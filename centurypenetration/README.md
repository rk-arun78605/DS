# Century Stock Penetration Dashboard - Database Strategy

## Overview
Optimized database architecture for analyzing CENTURY brand stock penetration across shops with real-time metrics for stock positioning, sales trends, and replenishment requirements.

## Database Architecture

### **Strategy Highlights**
1. **Partitioned Sales Table** - Monthly partitions for fast date-range queries
2. **Materialized Views** - Pre-aggregated metrics for instant dashboard loading
3. **Smart Indexing** - Brand-specific and composite indexes for optimal filtering
4. **CENTURY-Only Filter** - All views filtered to CENTURY brand at database level
5. **Daily Refresh Pattern** - Materialized views refresh once daily or on-demand

---

## Tables

### 1. **reorder_level** (Master Stock Data)
Stores current inventory, reorder parameters, and item master data.

**Key Columns:**
- `item_code`, `shop_code` - Primary key
- `shop_stock` - Stock In Hand (SIH)
- `brand` - Filtered to 'CENTURY'
- `min_nu`, `max_nu`, `reorder_qty` - Reorder parameters
- `selling_price` - Current selling price

**Data Source:** `GEN_reorder_till10dec25.csv`

**Performance Features:**
- Partial indexes for CENTURY brand only
- Composite indexes for common query patterns

---

### 2. **sales** (Transaction History - PARTITIONED)
Daily sales transactions partitioned by month for optimal performance.

**Key Columns:**
- `shop_code`, `item_code`, `date_invoice` - Primary key
- `qty` - Quantity sold
- `net_sales` - Sales value
- `dept`, `groups`, `sub_group` - Product classification

**Data Source:** Monthly CSV files (`jan25.csv`, `feb25.csv`, etc.)

**Performance Features:**
- 12 monthly partitions (Jan-Dec 2025)
- Partition pruning eliminates scanning irrelevant months
- Indexes on each partition for fast lookups

---

### 3. **sit** (Stock In Transit)
Items currently in transit from warehouse to shops.

**Key Columns:**
- `shop_code`, `item_code`, `dt_trans_date` - Primary key
- `nu_transit_qty` - Quantity in transit

**Data Source:** `GEN_SIT_till10dec25.csv`

---

## Materialized Views (Pre-Calculated Metrics)

### 1. **mv_sales_metrics**
Pre-aggregated sales for 30, 60, 90, 365 days per item per shop.

**Calculated Metrics:**
- `sales_30d`, `sales_60d`, `sales_90d`, `sales_365d` - Quantity sold
- `value_30d`, `value_60d`, `value_90d`, `value_365d` - Sales value
- `ros` - Rate of Sales (90-day average: `sales_90d / 90`)
- `last_sale_date` - Most recent sale

**Refresh:** Daily or on-demand

---

### 2. **mv_sit_summary**
Aggregated Stock In Transit per item per shop.

**Metrics:**
- `total_sit` - Total quantity in transit
- `latest_transit_date` - Most recent transit
- `transit_count` - Number of transit batches

**Refresh:** Daily

---

### 3. **mv_century_penetration** (Main Dashboard View)
Complete analytical view combining all metrics for CENTURY brand.

**All Metrics:**
- **Stock Levels:** `sih`, `sit`, `total_stock` (SIH + SIT)
- **Sales:** 30/60/90/365 day quantities and values
- **Rate of Sales:** `ros` (90-day daily average)
- **Requirement:** `req_21_days` (ROS × 21)
- **Stock Variance:** `total_stock - req_21_days`
- **Stock Status:** 
  - `OverStock` if variance > 0
  - `UnderStock` if variance < 0
  - `Balanced` if variance = 0
- **Days of Stock:** `total_stock / ros`

**Refresh:** Daily or on-demand

---

## Setup Instructions

### Step 1: Create Database and Tables
```bash
# Connect to PostgreSQL
psql -U postgres -p 3307 -h localhost

# Create database and schema
\i create_century_tables.sql
```

### Step 2: Load Data

**A. Reorder Level Data (Ready to Load)**
```bash
# Ensure file exists: GEN_reorder_till10dec25.csv
python load_century_data.py
```

**B. SIT Data (Ready to Load)**
```bash
# Ensure file exists: GEN_SIT_till10dec25.csv
python load_century_data.py
```

**C. Sales Data (Load when files ready)**
Place monthly CSV files in `D:\Dashboard Code\NO_WH\DS\centurypenetration\`:
- `jan25.csv`
- `feb25.csv`
- `mar25.csv`
- ... (through `dec25.csv`)

Then run:
```bash
python load_century_data.py
```

The script will:
- Auto-detect available files
- Skip missing months
- Load only found files
- Log results to `data_load_log.txt`

---

## Daily Maintenance

### Refresh Materialized Views
Run daily after data updates:

```sql
-- Option 1: SQL function
SELECT refresh_century_views();

-- Option 2: Manual refresh
REFRESH MATERIALIZED VIEW CONCURRENTLY mv_sales_metrics;
REFRESH MATERIALIZED VIEW CONCURRENTLY mv_sit_summary;
REFRESH MATERIALIZED VIEW CONCURRENTLY mv_century_penetration;
```

**Schedule via Windows Task Scheduler:**
```bash
# Create batch file: refresh_century_views.bat
psql -U postgres -p 3307 -d century_penetration -c "SELECT refresh_century_views();"

# Schedule daily at 6 AM
```

---

## Dashboard Queries

### 1. Overview Metrics
```sql
SELECT 
    COUNT(DISTINCT item_code) as total_items,
    COUNT(DISTINCT shop_code) as total_shops,
    SUM(CASE WHEN stock_status = 'UnderStock' THEN 1 ELSE 0 END) as understock_items,
    SUM(CASE WHEN stock_status = 'OverStock' THEN 1 ELSE 0 END) as overstock_items,
    SUM(CASE WHEN stock_status = 'Balanced' THEN 1 ELSE 0 END) as balanced_items,
    SUM(sih) as total_sih,
    SUM(sit) as total_sit,
    ROUND(SUM(req_21_days), 0) as total_requirement
FROM mv_century_penetration;
```

### 2. Top 20 Understock Items
```sql
SELECT 
    item_code,
    item_name,
    shop_code,
    sih,
    sit,
    total_stock,
    ros,
    req_21_days,
    stock_variance,
    days_of_stock
FROM mv_century_penetration
WHERE stock_status = 'UnderStock'
ORDER BY stock_variance ASC
LIMIT 20;
```

### 3. Shop-wise Stock Status
```sql
SELECT 
    shop_code,
    COUNT(*) as total_items,
    SUM(CASE WHEN stock_status = 'UnderStock' THEN 1 ELSE 0 END) as understock,
    SUM(CASE WHEN stock_status = 'OverStock' THEN 1 ELSE 0 END) as overstock,
    SUM(CASE WHEN stock_status = 'Balanced' THEN 1 ELSE 0 END) as balanced,
    ROUND(SUM(sih), 0) as total_sih,
    ROUND(SUM(sit), 0) as total_sit,
    ROUND(AVG(ros), 2) as avg_ros
FROM mv_century_penetration
GROUP BY shop_code
ORDER BY understock DESC;
```

### 4. Department-wise Analysis
```sql
SELECT 
    dept,
    COUNT(DISTINCT item_code) as total_items,
    SUM(CASE WHEN stock_status = 'UnderStock' THEN 1 ELSE 0 END) as understock,
    SUM(sih) as total_sih,
    SUM(sit) as total_sit,
    ROUND(SUM(req_21_days), 0) as total_requirement,
    ROUND(AVG(ros), 2) as avg_ros
FROM mv_century_penetration
GROUP BY dept
ORDER BY understock DESC;
```

### 5. Slow Moving Items (Low ROS)
```sql
SELECT 
    item_code,
    item_name,
    shop_code,
    dept,
    sih,
    sit,
    ros,
    days_of_stock,
    last_sale_date
FROM mv_century_penetration
WHERE ros < 0.5  -- Less than 0.5 units per day
  AND sih > 0
ORDER BY ros ASC, days_of_stock DESC
LIMIT 50;
```

### 6. Critical Understock (No Stock, High Demand)
```sql
SELECT 
    item_code,
    item_name,
    shop_code,
    dept,
    sih,
    sit,
    total_stock,
    ros,
    req_21_days,
    stock_variance,
    sales_30d,
    sales_90d
FROM mv_century_penetration
WHERE total_stock = 0
  AND ros > 1  -- At least 1 unit per day demand
ORDER BY ros DESC
LIMIT 50;
```

---

## Performance Optimization

### 1. Indexes Created
- **CENTURY brand filter** - Partial index on reorder_level
- **Composite indexes** - (item_code, shop_code) on all tables
- **Date indexes** - DESC order for recent-first queries
- **Status indexes** - Partial indexes on UnderStock/OverStock

### 2. Query Performance Tips
- Always query from `mv_century_penetration` for dashboard
- Use materialized views instead of joining base tables
- Refresh views daily, not on every query
- Filter by shop or dept for faster results

### 3. Maintenance Commands
```sql
-- Analyze tables after bulk load
ANALYZE reorder_level;
ANALYZE sales;
ANALYZE sit;

-- Vacuum if needed (after deletes/updates)
VACUUM ANALYZE reorder_level;

-- Check materialized view sizes
SELECT 
    schemaname,
    matviewname,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||matviewname)) as size
FROM pg_matviews
WHERE matviewname LIKE 'mv_%';
```

---

## Data Flow Diagram

```
CSV Files (Data Sources)
    |
    ├── GEN_reorder_till10dec25.csv ──> reorder_level (Master)
    ├── GEN_SIT_till10dec25.csv ──────> sit (Transit)
    └── Monthly CSVs (jan25-dec25) ───> sales (Partitioned)
                |
                v
    [ Base Tables with Indexes ]
                |
                v
    [ Nightly/On-Demand Refresh ]
                |
                v
    ┌─────────────────────────────┐
    │  Materialized Views         │
    ├─────────────────────────────┤
    │  mv_sales_metrics           │  <- Pre-aggregated sales
    │  mv_sit_summary             │  <- Aggregated transit
    │  mv_century_penetration     │  <- Complete dashboard view
    └─────────────────────────────┘
                |
                v
    [ Dashboard Queries (Fast!) ]
```

---

## File Structure

```
D:\Dashboard Code\NO_WH\DS\centurypenetration\
│
├── create_century_tables.sql      # Database schema and views
├── load_century_data.py            # Data loading script
├── data_load_log.txt               # Loading logs
│
├── GEN_reorder_till10dec25.csv     # Reorder level data
├── GEN_SIT_till10dec25.csv         # Stock in transit data
│
└── Sales CSV files (when ready):
    ├── jan25.csv
    ├── feb25.csv
    ├── mar25.csv
    ├── ... (through dec25.csv)
```

---

## Key Formulas

| Metric | Formula | Description |
|--------|---------|-------------|
| **SIH** | `shop_stock` | Stock In Hand (current inventory) |
| **SIT** | `SUM(nu_transit_qty)` | Stock In Transit (from sit table) |
| **Total Stock** | `SIH + SIT` | Available + incoming stock |
| **ROS** | `sales_90d / 90` | Rate of Sales (daily average) |
| **Req 21 Days** | `ROS × 21` | Required stock for 21 days |
| **Stock Variance** | `(SIH + SIT) - Req 21 Days` | Surplus or shortage |
| **Days of Stock** | `Total Stock / ROS` | Days until stockout |

---

## Next Steps

1. ✅ Run `create_century_tables.sql` to create schema
2. ✅ Place CSV files in `centurypenetration` folder
3. ✅ Run `load_century_data.py` to load reorder and SIT data
4. ⏳ Add sales CSV files when ready
5. ⏳ Run `load_century_data.py` again to load sales
6. ✅ Query `mv_century_penetration` for dashboard
7. ✅ Schedule daily view refresh

---

## Support

For questions or issues:
- Check `data_load_log.txt` for loading errors
- Verify CSV file formats match column mappings
- Ensure PostgreSQL is running on port 3307
- Confirm CENTURY brand exists in reorder_level data
