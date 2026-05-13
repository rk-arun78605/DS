# Melcom Retail Analytics Platform - AI Agent Guide

## System Architecture Overview

**Home Portal** (`home_dashboard.py`, port 8501) serves as the central hub for 5 independent Streamlit applications, all sharing PostgreSQL infrastructure on port **3307**:

1. **Serial Tracking** (`SerialNoReport/serial_funnel_dashboard.py`, port 8502) - Supply chain funnel for serialized items
2. **KPI Dashboard** (`kpi_app/kpi_dashboard.py`, port 8503) - Multi-page sales analytics with year-over-year comparisons
3. **NO_WH Inventory Pulse** (`nowhstock/nowhstock_ds_FINAL.py`, port 8504) - Stock transfer recommendations via materialized views
4. **Barcode Matcher** (`barcode_matcher/barcode_matcher_app.py`, port 8505) - Product matching system
5. **Century Penetration** (`centuryPenetration.py`, port 8506) - Market penetration analysis with LLM-powered insights

**Launch all dashboards**: Run `start_all_dashboards.bat` (starts all 6 windows in parallel, auto-opens home portal after 10s)

## Core Database Pattern (CRITICAL)

**All apps use identical connection architecture:**
```python
@st.cache_resource
def get_connection_pool(dbname: str):
    return psycopg2.pool.SimpleConnectionPool(
        minconn=1, maxconn=10,  # KPI uses 5-20 for heavy load
        host='localhost', port=3307, user='postgres', password='hello',
        dbname=dbname
    )

@contextmanager
def get_db_connection(dbname: str):
    pool = get_connection_pool(dbname)
    conn = pool.getconn()
    try:
        yield conn
    finally:
        pool.putconn(conn)

# Usage (ALWAYS use context managers, NEVER hold connections)
with get_db_connection('salesdata') as conn:
    with conn.cursor(cursor_factory=RealDictCursor) as cursor:
        cursor.execute(query, params)
        # Connection auto-returned to pool after block
```

**Database segregation:**
- `users` - Authentication (all apps, shared login system)
- `salesdata` - Sales history (`sales_2024`, `sales_2025` by year), inventory, GRN data, materialized views
- `pidashboard` - Physical inventory transactions, variance tracking
- `serial_tracking` - Serialized item lifecycle (loaded → offloaded → sold)

## Home Portal Hub (`home_dashboard.py`)

**Central landing page (port 8501) with:**
- **IPv4 Auto-Detection**: Automatically detects machine's IP address for network sharing
- **Live Status Monitoring**: Real-time checks if each dashboard is online (polls port 8502-8506)
- **One-Click Launch**: Buttons to open each dashboard; disabled if offline
- **Responsive Grid**: 2-3 columns based on screen size
- **Custom CSS**: Purple gradient background with hover effects

**Key implementation patterns:**
```python
# IPv4 detection and dashboard status check
import socket
def get_ipv4():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))
    ip = s.getsockname()[0]
    s.close()
    return ip

# Check if dashboard is running on a port
def is_dashboard_online(port):
    try:
        response = requests.get(f'http://localhost:{port}', timeout=2)
        return response.status_code == 200
    except:
        return False
```

**Dashboard Configuration (see `HOME_DASHBOARD_README.md`):**
- Serial Tracking: port 8502, file = `SerialNoReport/serial_funnel_dashboard.py`
- KPI Dashboard: port 8503, file = `kpi_app/kpi_dashboard.py`
- NO_WH Inventory: port 8504, file = `nowhstock/nowhstock_ds_FINAL.py`
- Barcode Matcher: port 8505, file = `barcode_matcher/barcode_matcher_app.py`
- Century Penetration: port 8506, file = `centuryPenetration.py`

## Materialized View Strategy (NO_WH Core Architecture)

**ALL business logic lives in SQL** (`NowhStock_mv_recommendations_complete.sql`):
- 489-line CTE with complete recommendation engine
- Priority shop allocation, expiry checks, demand capping, FEFO ordering
- Python just queries: `SELECT * FROM mv_recommendations_complete WHERE [filters]`
- Performance: <100ms (vs 30+ seconds in pure Python)

**Critical business rules enforced in view:**
- Source: Non-priority shops with stock > 30d sales
- Destinations: 11 priority shops only (SPN, MSS, LFS, M03, KAS, MM1, MM2, FAR, KS7, WHL, MM3)
- Cap: `GREATEST(dest_wh_grn_30d_sales, dest_sales_30d)` per item+destination
- Expiry: Block transfers if source item < 30 days to expiry
- Allocation: Window function with cumulative SUM, never exceed cap
- Sorting: Oldest GRN first (FEFO), then by priority rank

**View refresh workflow:**
```powershell
# Python wrapper (recommended - logs to file)
python nowhstock\batch\refresh_all_views.py

# Direct SQL (faster for emergency refresh)
psql -U postgres -d salesdata -p 3307 -f refresh_all_views.sql

# Scheduled refresh (Windows Task Scheduler)
.\nowhstock\batch\refresh_views_scheduled.bat
```

## Streamlit Caching Hierarchy (Performance Critical)

**Cache decorators by data lifetime:**
```python
@st.cache_resource  # Connection pools ONLY (never expires, process-global)
def get_connection_pool(dbname: str): ...

@st.cache_data(ttl=3600)  # Static data: filter options, metadata (1hr)
def load_filter_options(): ...

@st.cache_data(ttl=600)  # Sales data, KPIs (10min, balances freshness)
def load_sales_data(): ...

@st.cache_data(ttl=300, show_spinner=False)  # Auth lookups (5min, silent)
def authenticate_user(employee_id, password): ...
```

**KPI Dashboard instant filtering pattern:**
```python
@st.cache_data(ttl=3600)
def preload_all_months():
    """Load all 11 months at startup (750KB cached)"""
    return {month: get_month_data(month) for month in MONTHS}

# Result: First click 0.5s, subsequent clicks 0ms (instant from cache)
```

**Session state pattern (ALL apps):**
```python
# Initialize at app start (before any st.* calls)
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'user' not in st.session_state:
    st.session_state.user = None

# Authentication check
def check_table_access(user: Dict, required_table: str) -> bool:
    user_tables = [t.strip().lower() for t in user.get('table_access', '').split(',')]
    return required_table.lower() in user_tables or 'all' in user_tables
```

## Date Handling Convention (STRICT - ALL APPS)

**"Yesterday as end date" business rule:**
```python
# Rationale: Today's data incomplete until EOD batch runs
today = datetime.today()
yesterday = today - timedelta(days=1)  # Always use yesterday as latest

# MTD: First day of current month → yesterday (inclusive)
first_day = yesterday.replace(day=1)
mtd_start, mtd_end = first_day, yesterday

# 30-day rolling window: yesterday - 29 days → yesterday (30 days inclusive)
start_date = yesterday - timedelta(days=29)
end_date = yesterday

# Year-over-year: Same calendar dates, different years
start_2024 = start_date.replace(year=2024)
end_2024 = end_date.replace(year=2024)
start_2025 = start_date.replace(year=2025)
end_2025 = end_date.replace(year=2025)
```

**Dynamic year handling (handles Dec→Jan overlaps):**
```python
# Sales queries must handle year boundaries (e.g., Dec 20 + 30 days → Jan 19)
years_to_query = list(range(start_date.year, end_date.year + 1))
for year in years_to_query:
    # Query sales_{year} table for date range
```

**Excel export fix (REQUIRED):**
```python
# Remove timezone to prevent Excel corruption
df['date_column'] = df['date_column'].dt.tz_localize(None)
```

## Multi-Page App Structure (KPI Dashboard)

```
kpi_app/
├── kpi_dashboard.py          # Main page (overview, shop rankings, MTD/YTD)
└── pages/
    ├── 1_📊_Department_Analysis.py   # Dept-level drilldown
    ├── 2_📁_Group_Analysis.py        # Product group analysis
    ├── 3_📑_SubGroup_Analysis.py     # SubGroup breakdown
    └── 4_🏪_Shop_Analysis.py         # Shop-level performance
```
- **Naming:** Prefix `N_` for sort order, emoji in filename shows in sidebar
- **Shared state:** Use `st.session_state` for cross-page filters/user data
- **Run:** `streamlit run kpi_app\kpi_dashboard.py` (auto-discovers pages/)
- **Currently:** Pages disabled (moved to `pages_disabled/`) - all analysis on main page

## Business Logic (NO_WH Inventory)

**Priority shop allocation (hardcoded, ORDER MATTERS):**
```python
PRIORITY_SHOPS = ['SPN', 'MSS', 'LFS', 'M03', 'KAS', 'MM1', 'MM2', 'FAR', 'KS7', 'WHL', 'MM3']
# Rank 1-11: SPN gets first allocation, MM3 gets remaining capacity
```

**Critical concepts:**
- **WH GRN Date**: When warehouse received item from supplier (item-level, same for all shops)
- **Shop GRN Date**: When individual shop received item from warehouse (shop-specific)
- **WH GRN +30d Sales**: Sales from WH GRN date to +30 days (captures post-arrival demand spike)
- **Destination Cap**: `MAX(dest_sales_30d, dest_wh_grn_30d_sales)` - uses higher demand signal

**Blocking rules (see `NowhStock_mv_recommendations_complete.sql` line 400+):**
1. Same shop transfers: `source_shop != dest_shop`
2. Priority shops as sources: Never (priority shops only receive)
3. Expiry check: Block if `source_expiry_days < 30` (30-day safety buffer)
4. No sales demand: Block if all three sales metrics = 0
5. Cumulative cap: Running SUM across sources never exceeds destination cap

**Expiry handling (FEFO - First Expiry First Out):**
- Expiry data from `shopexpiry` table (latest date per item+shop)
- Sort allocation: Oldest GRN age + earliest expiry first
- Status labels: `< 30 days = "Expiring"`, `>= 30 days = "Safe"`, `NULL = "No Expiry"`

**Config class pattern (centralized constants):**
```python
class Config:
    DB_CONFIG = {'host': 'localhost', 'user': 'postgres', 'password': 'hello', 'port': 3307}
    PRIORITY_SHOPS = ['SPN', 'MSS', 'LFS', ...]  # Order = allocation priority
    DEFAULT_THRESHOLD = 30  # Fast-moving items (≥30 sales in 30d)
    BUFFER_DAYS = 30  # Stock cover target days
    GRN_FALLBACK_DAYS = 90  # If no GRN date, assume 90 days old
    PAGE_TITLE = "Inventory Pulse NO_WH"
    LOGO_URL = "https://melcom.com/media/favicon/stores/1/faviconn_162_x_184px.jpg"
```

## Performance Optimization

**Index strategy (see `kpi_app/INDEX_ANALYSIS.md`):**
- **Keep:** Date indexes only (`idx_sales_2024_date_invoice_date`, `idx_sales_2025_date_invoice_date`)
- **Drop:** 33 unused composite indexes (0 scans, 5.5GB wasted)
- **Why unused:** Queries don't match column order, PostgreSQL prefers simpler indexes

**Query patterns:**
```python
# ✅ GOOD: Single CTE with FULL OUTER JOIN for 2024 vs 2025
WITH sales_2024 AS (...), sales_2025 AS (...)
SELECT COALESCE(s24.shop, s25.shop) AS shop, s24.sales, s25.sales
FROM sales_2024 s24 FULL OUTER JOIN sales_2025 s25 USING (shop, dept)

# ❌ BAD: Separate queries then pandas merge (2x round-trips)
df_2024 = pd.read_sql("SELECT * FROM sales_2024 WHERE ...", conn)
df_2025 = pd.read_sql("SELECT * FROM sales_2025 WHERE ...", conn)
df = df_2024.merge(df_2025, ...)  # Slow
```

**Pandas optimization:**
```python
# ❌ BAD: apply() with lambda (row-by-row, slow)
df['result'] = df.apply(lambda row: lookup_dict.get(row['key']), axis=1)

# ✅ GOOD: Vectorized map() (10x faster)
df['result'] = df['key'].map(lookup_dict)

# ✅ GOOD: Pre-compute lookup dict from second dataframe
lookup = df2.set_index('key')['value'].to_dict()
df['result'] = df['key'].map(lookup)
```

## Data Loading & Maintenance

**Batch data load (see `nowhstock/batch/load_all_data.py`):**
```powershell
# Load all CSV files into PostgreSQL tables
cd nowhstock\batch
python load_all_data.py  # Progress tracking, validation, logging

# Tables loaded: inventory_master, sales_2024/2025, sup_shop_grn, shopexpiry, itemdetails
```

**View refresh after data changes:**
```powershell
# Step 1: Load new data
python nowhstock\batch\load_all_data.py

# Step 2: Refresh materialized views (CRITICAL)
python nowhstock\batch\refresh_all_views.py

# Step 3: Verify view freshness
psql -U postgres -d salesdata -p 3307 -c "SELECT matviewname, last_refresh FROM pg_matviews WHERE matviewname LIKE 'mv_%';"
```

## Web Scrapers (Competitor Pricing)

**Fareway** (`fareway.py`):
- Target: `fairwayghana.com`
- Concurrency: 20 threads, 20s timeout
- Pagination: 5 pages per category
- Output: `Fareway_Products_Output.xlsx`

**MaxMart** (`new script_MAXMART.py`):
- Target: `maxmartonline.com`
- Concurrency: 25 threads, 15s timeout, `verify=False` (SSL disabled)
- Pagination: 50 pages per category (`?pagenumber={page}&viewmode=grid`)
- Output: `MaxMart_Products_Output_Fast.xlsx`

**Shared scraper pattern:**
```python
from concurrent.futures import ThreadPoolExecutor
from bs4 import BeautifulSoup

def scrape_page(url):
    response = requests.get(url, timeout=20)
    soup = BeautifulSoup(response.text, 'html.parser')
    # Extract: Product, Price, Breadcrumb, URL
    return parsed_data

# Concurrent execution
with ThreadPoolExecutor(max_workers=25) as executor:
    futures = {executor.submit(scrape_page, url): url for url in all_urls}
    for future in as_completed(futures):
        results.append(future.result())
```

## Running Applications

**Option 1: Start All Dashboards via Batch Script (RECOMMENDED)**
```powershell
# Windows: Runs all 5 dashboards + home portal in parallel
.\start_all_dashboards.bat

# Then access: http://{YOUR_IPv4}:8501 (auto-opens in Home Portal)
```

**Option 2: Start Individual Dashboards**
```powershell
# Home Portal (central hub with status monitoring)
streamlit run home_dashboard.py --logger.level=error

# NO_WH Inventory Dashboard (main recommendation system)
streamlit run nowhstock\nowhstock_ds_FINAL.py --server.port 8504

# KPI Dashboard (multi-page sales analytics)
streamlit run kpi_app\kpi_dashboard.py --server.port 8503

# Serial Tracking (supply chain funnel)
streamlit run SerialNoReport\serial_funnel_dashboard.py --server.port 8502

# Barcode Matcher (product matching)
streamlit run barcode_matcher\barcode_matcher_app.py --server.port 8505

# Century Penetration (market penetration analysis)
streamlit run centuryPenetration.py --server.port 8506
```

**Batch Maintenance Scripts**
```powershell
# Data load
python nowhstock\batch\load_all_data.py        # Load all CSV files

# View refresh (CRITICAL after data changes)
python nowhstock\batch\refresh_all_views.py    # Refresh materialized views

# Daily sales upload
python nowhstock\batch\daily_sales_upload.py   # Update sales_2024/2025

# Competitor price scrapers
python fareway.py                              # Fareway Ghana pricing
python "new script_MAXMART.py"                 # MaxMart pricing
```

## Debugging & Troubleshooting

**Check materialized view status:**
```sql
SELECT matviewname, last_refresh, pg_size_pretty(pg_total_relation_size(schemaname||'.'||matviewname))
FROM pg_matviews 
WHERE matviewname LIKE 'mv_%';
```

**Common issues:**
- `mv_recommendations_complete` doesn't exist → Run `NowhStock_mv_recommendations_complete.sql`
- Slow queries → Check `INDEX_ANALYSIS.md`, run `ANALYZE` on base tables
- Cache issues → Clear with `st.cache_data.clear()` or restart app
- View data stale → Run `python nowhstock\batch\refresh_all_views.py`
- Zero recommendations → Check expiry blocking (< 30 days), ensure view refreshed after data load

**Authentication pattern:**
```python
# String comparison (database stores lowercase 'true')
user_tables = user['table_access'].lower().split(',')
is_active = user['is_active'] == 'true'
has_access = required_table in user_tables or 'all' in user_tables
```

**Logging pattern (NO_WH app):**
```python
import logging
logger = logging.getLogger(__name__)
logger.info(f"✅ Loaded {len(df)} recommendations")
logger.error(f"❌ Cap violation: {item} → {dest}")
```
## PostgreSQL Column Case Sensitivity (CRITICAL for Data Uploads)

**Key Rules:**
- Unquoted identifiers → PostgreSQL stores as **lowercase**: `SHOP_CODE` becomes `shop_code`
- Double-quoted identifiers → PostgreSQL stores exact case: `"SHOP_CODE"` stays `"SHOP_CODE"`
- Column references in SQL must match exact case (case-sensitive)
- COPY commands must reference columns by exact case as they exist in table

**Common Error & Solution:**
```
ERROR: column "SHOP_CODE" of relation "sales_2026" does not exist
```

**Fix for home_dashboard.py uploads:**
1. Use `DROP TABLE` (not `TRUNCATE`) to remove old schema and force recreation with correct case
2. Apply column mapping for COPY commands - convert config uppercase to table lowercase:
```python
col_map = {
    'SHOP_CODE': 'shop_code',       # Config uppercase → DB lowercase
    'ITEM_CODE': 'item_code',
    'DATE_INVOICE': 'date_invoice'
}
copy_cols = [col_map.get(col, col.lower()) for col in config_columns]
copy_sql = f"COPY {table_name} ({', '.join(copy_cols)}) FROM STDIN WITH CSV"
```

**Why this matters:**
- CSV config uses uppercase column names (convention)
- PostgreSQL defaults to lowercase storage (default behavior)
- COPY must match exact column case as stored in table
- **Prevention:** Document expected column case in TABLE_CONFIGS

## LLM Integration Pattern (Century Penetration Dashboard)

**Multi-provider LLM setup with automatic fallback:**
```python
# Three LLM providers via OpenAI-compatible API (see testing_openai/test_centurypenetration_openai.py)
openai_client = OpenAI(api_key=OPENAI_API_KEY)

gemini_client = OpenAI(
    api_key=GEMINI_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

pplx_client = OpenAI(
    api_key=PERPLEXITY_API_KEY,
    base_url="https://api.perplexity.ai"
)

# Automatic fallback logic (preferred → fallback1 → fallback2)
def ask_llm_any(user_prompt: str, preferred: str = "Perplexity"):
    """Try preferred provider first, fallback to others on failure"""
    providers = ["Perplexity", "OpenAI", "Gemini"]  # Ordered by preference
    for provider in providers:
        try:
            if provider == "OpenAI" and openai_client:
                return call_with_client(openai_client, "gpt-4o-mini", ...)
            # ... similar for Gemini and Perplexity
        except RateLimitError:
            time.sleep(1)
            continue  # Try next provider
        except (APIError, AuthenticationError):
            continue  # Try next provider
```

**Key configuration (testing_openai/.streamlit/secrets.toml):**
```toml
[llm]
openai_key = "sk-proj-..."
gemini_key = "AIza..."
perplexity_key = "pplx-..."
```

**Models used:**
- OpenAI: `gpt-4o-mini` (fast, cost-effective)
- Gemini: `gemini-3-flash-preview` (via OpenAI-compatible endpoint)
- Perplexity: `llama-3.1-sonar-small-128k-chat` (web-search enabled)

## Staging Environment (Priority-to-Priority Testing)

**Purpose:** Test new business logic (priority shops can be sources) without touching production

**Location:** `nowhstock/staging/`

**Key files:**
- `nowhstock_ds_STAGING.py` - Staging dashboard (uses staging views)
- `sql/create_staging_views.sql` - All staging materialized views
- `batch/1_setup_staging.bat` - Create staging environment (5-10 min)
- `batch/2_run_staging_dashboard.bat` - Launch staging dashboard
- `batch/3_refresh_staging_views.bat` - Refresh staging views with latest data
- `batch/5_promote_to_production.bat` - Promote staging to production (AFTER APPROVAL)

**Naming convention:** All staging objects have `_staging` suffix:
- `mv_recommendations_complete_staging` (vs `mv_recommendations_complete`)
- `mv_slow_fast_moving_summary_staging` (vs `mv_slow_fast_moving_summary`)

**Workflow:**
```powershell
# 1. Create staging environment
cd nowhstock\staging\batch
1_setup_staging.bat  # Creates staging views

# 2. Test staging dashboard
2_run_staging_dashboard.bat  # Runs on port 8502

# 3. Compare results side-by-side
4_compare_staging_vs_production.bat  # Shows differences

# 4. Promote to production (only after approval)
5_promote_to_production.bat  # Replaces production views
```

## Performance & Scalability Patterns

**Auto-LIMIT pattern (prevents memory crashes):**
```python
# NO_WH dashboard: Auto-apply LIMIT when no filters (see DASHBOARD_PERFORMANCE_FIX.md)
if all_filters_are_all():
    # No filters = 500K+ rows (399 MB) → CRASH
    query += " LIMIT 50000"  # Safe: 50K rows (45 MB), loads in 6-8s

if user_checked_generate_all:
    query = query.replace("LIMIT 50000", "")  # User explicitly wants all

# Results:
# - No filters: 50K rows, 6-8s, 45 MB ✅
# - Shop filter: 1,500 rows, 0.5s, 2 MB ✅
# - Single product: 50 rows, 0.1s, 0.1 MB ✅
```

**Materialized view allocation logic (see MATERIALIZED_VIEW_FIXES.md):**
```sql
-- CRITICAL: Window function must partition by BOTH item AND destination
-- Each destination gets cap allocation independently
SUM(allocated_qty) OVER (
    PARTITION BY item_code, dest_shop  -- ✅ CORRECT (not just item_code)
    ORDER BY priority_rank, source_grn_age DESC
) AS cumulative_allocation

-- Example: LFS cap = 41 units
-- ❌ WRONG: Each source gives 41 → 10 sources = 410 units (10× overstock)
-- ✅ CORRECT: All sources combined give 41 total
```

**Index strategy updates:**
- **Sales tables:** Only date indexes needed (`idx_sales_2024_date_invoice_date`, `idx_sales_2025_date_invoice_date`)
- **Barcode matcher:** Partial index on `is_active = true` (90% reduction in index size)
- **Century penetration:** Monthly partitions with partition pruning (eliminates scanning irrelevant months)
- **Drop unused composite indexes:** See `INDEXES_USED.md` (33 unused indexes = 5.5GB wasted)

## Recent Critical Fixes (Reference Documentation)

**Sales 2026 column case issue:** See `FIX_SALES_2026_COLUMN_CASE.md`
- Use `DROP TABLE` (not `TRUNCATE`) to reset schema
- Apply column mapping for COPY commands (uppercase config → lowercase table)

**Item overallocation bug:** See `MATERIALIZED_VIEW_FIXES.md`
- Fixed window function to partition by both `item_code` AND `dest_shop`
- Changed expiry threshold from `< 0` to `< 30` days (30-day safety buffer)

**Dashboard memory crash:** See `DASHBOARD_PERFORMANCE_FIX.md`
- Auto-apply LIMIT 50,000 when no filters
- Reduced data size from 399 MB (crash) to 45 MB (6-8s load)

**Month partition fix:** See `MONTH_PARTITION_FIX_SUMMARY.md`
- Century Penetration uses 12 monthly partitions (Jan-Dec 2025)
- Partition pruning automatically eliminates irrelevant months from queries