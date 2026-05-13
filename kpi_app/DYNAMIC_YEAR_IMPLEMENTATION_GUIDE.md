# KPI Dashboard Dynamic Year Support - Implementation Guide

## Summary of Changes

This document outlines all changes needed to make the KPI dashboard automatically detect and use available sales tables (sales_2024, sales_2025, sales_2026, etc.) without hardcoded years.

## 1. Core Functions Added (Already Implemented)

```python
@st.cache_data(ttl=3600)
def get_available_years():
    """Dynamically detect available sales tables from database"""
    # Queries pg_tables for sales_YYYY tables
    # Returns: [2024, 2025, 2026, ...]

@st.cache_data(ttl=3600)
def get_current_year():
    """Get current year"""
    return datetime.now().year

@st.cache_data(ttl=3600)
def get_previous_year():
    """Get previous year for YoY comparisons"""
    return get_current_year() - 1

@st.cache_data(ttl=3600)
def get_year_range():
    """Get (previous_year, current_year) tuple for comparisons"""
    # Returns: (2025, 2026) if running in 2026
```

## 2. Pattern for Updating Query Functions

### OLD Pattern (Hardcoded):
```python
query = """
    WITH data_2024 AS (
        SELECT * FROM sales_2024 WHERE ...
    ),
    data_2025 AS (
        SELECT * FROM sales_2025 WHERE ...
    )
    SELECT 
        d24.net_sales as net_sales_2024,
        d25.net_sales as net_sales_2025
    FROM data_2024 d24
    FULL OUTER JOIN data_2025 d25 ...
"""
```

### NEW Pattern (Dynamic):
```python
prev_year, curr_year = get_year_range()

query = f"""
    WITH data_prev AS (
        SELECT * FROM sales_{prev_year} WHERE ...
    ),
    data_curr AS (
        SELECT * FROM sales_{curr_year} WHERE ...
    )
    SELECT 
        dp.net_sales as net_sales_{prev_year},
        dc.net_sales as net_sales_{curr_year}
    FROM data_prev dp
    FULL OUTER JOIN data_curr dc ...
"""

# Rename columns for backward compatibility with existing UI code
df = df.rename(columns={
    f'net_sales_{prev_year}': 'net_sales_2024',
    f'net_sales_{curr_year}': 'net_sales_2025'
})
```

## 3. Functions That Need Updates

### ✅ Already Updated:
1. `get_dept_mtd_data()` - MTD department data

### ⚠️ Need Update (Same Pattern):
1. `get_dept_ytd_data()` - YTD department data
2. `get_shop_mtd_data()` - MTD shop data
3. `get_shop_ytd_data()` - YTD shop data
4. `get_monthly_trend_data()` - Monthly trends
5. `get_filtered_month_data()` - Month filtering
6. `preload_all_months()` - Month preloading
7. Any other functions with `sales_2024` or `sales_2025` in queries

## 4. Display Label Updates

### In main() function, update year labels dynamically:

```python
# OLD:
st.subheader("2024 vs 2025 Comparison")

# NEW:
prev_year, curr_year = get_year_range()
st.subheader(f"{prev_year} vs {curr_year} Comparison")
```

### Chart Labels:
```python
# OLD:
labels = ["2024", "2025"]

# NEW:
prev_year, curr_year = get_year_range()
labels = [str(prev_year), str(curr_year)]
```

## 5. Benefits of This Approach

1. **Auto-Detection**: Automatically discovers sales_YYYY tables in database
2. **Zero Maintenance**: Create `sales_2027` table → works immediately
3. **Backward Compatible**: Renames columns to 2024/2025 for existing UI code
4. **Fallback Safe**: Falls back to [2024, 2025, 2026] if detection fails
5. **Current Year Aware**: Always compares (current_year-1) vs (current_year)

## 6. Testing Checklist

- [ ] Create `sales_2027` table with sample data
- [ ] Restart dashboard - should auto-detect 2027
- [ ] Verify 2026 vs 2027 comparison shows in UI
- [ ] Check all charts show correct year labels
- [ ] Verify YoY growth calculations use correct years
- [ ] Test with missing year (e.g., no sales_2024) - should use available years

## 7. Future Enhancements

### Allow User Year Selection:
```python
available_years = get_available_years()
prev_year = st.selectbox("Compare Year", available_years[:-1], index=len(available_years)-2)
curr_year = st.selectbox("To Year", available_years, index=len(available_years)-1)
```

### Multi-Year Comparisons:
```python
selected_years = st.multiselect("Select Years", available_years, default=available_years[-2:])
# Build UNION ALL query for all selected years
```

## 8. Index Management

When creating new sales_YYYY table, remember to create indexes:

```sql
-- For sales_2027 (example)
CREATE INDEX idx_sales_2027_date_invoice_date ON sales_2027("DATE_INVOICE");
CREATE INDEX idx_sales_2027_shop_code ON sales_2027("SHOP_CODE");
CREATE INDEX idx_sales_2027_dept ON sales_2027("DEPT");
-- Add other indexes as needed
```

## 9. File Structure

```
kpi_app/
├── kpi_dashboard.py          # Main file (updated)
└── pages/                     # Multi-page files (need same updates)
    ├── 1_📊_Department_Analysis.py
    ├── 2_📁_Group_Analysis.py
    ├── 3_📑_SubGroup_Analysis.py
    └── 4_🏪_Shop_Analysis.py
```

Each page file needs the same year detection functions and query updates.

## 10. Quick Start Implementation

For each function with hardcoded years:

1. Add at top of function:
   ```python
   prev_year, curr_year = get_year_range()
   ```

2. Replace table names in query:
   ```python
   sales_2024 → sales_{prev_year}
   sales_2025 → sales_{curr_year}
   ```

3. Replace column names in query:
   ```python
   net_sales_2024 → net_sales_{prev_year}
   net_sales_2025 → net_sales_{curr_year}
   ```

4. Add column rename before return:
   ```python
   df = df.rename(columns={
       f'net_sales_{prev_year}': 'net_sales_2024',
       f'net_sales_{curr_year}': 'net_sales_2025',
       # ... other columns
   })
   ```

5. Update display labels in UI:
   ```python
   prev_year, curr_year = get_year_range()
   st.subheader(f"{prev_year} vs {curr_year} Performance")
   ```

## 11. Example: Complete Function Update

```python
@st.cache_data(ttl=3600)
def get_dept_ytd_data():
    """Get Department-wise YTD data - DYNAMIC YEAR SUPPORT"""
    start_date, end_date = get_ytd_dates()
    prev_year, curr_year = get_year_range()  # ← ADD THIS
    
    with get_sales_connection() as conn:
        # Build dynamic query
        query = f"""
            WITH data_prev AS (
                SELECT "DEPT", SUM("NET_SALES") as net_sales
                FROM sales_{prev_year}  -- ← DYNAMIC
                WHERE "DATE_INVOICE"::date >= %s AND "DATE_INVOICE"::date <= %s
                GROUP BY "DEPT"
            ),
            data_curr AS (
                SELECT "DEPT", SUM("NET_SALES") as net_sales
                FROM sales_{curr_year}  -- ← DYNAMIC
                WHERE "DATE_INVOICE"::date >= %s AND "DATE_INVOICE"::date <= %s
                GROUP BY "DEPT"
            )
            SELECT 
                COALESCE(dp."DEPT", dc."DEPT") as "DEPT",
                COALESCE(dp.net_sales, 0) as net_sales_{prev_year},  -- ← DYNAMIC
                COALESCE(dc.net_sales, 0) as net_sales_{curr_year}   -- ← DYNAMIC
            FROM data_prev dp
            FULL OUTER JOIN data_curr dc ON dp."DEPT" = dc."DEPT"
        """
        
        df = pd.read_sql(query, conn, params=(
            start_date.replace(year=prev_year),
            end_date.replace(year=prev_year),
            start_date.replace(year=curr_year),
            end_date.replace(year=curr_year)
        ))
    
    # Dynamic column calculations
    prev_col = f'net_sales_{prev_year}'
    curr_col = f'net_sales_{curr_year}'
    df['yoyg'] = ((df[curr_col] - df[prev_col]) / df[prev_col].replace(0, 1) * 100).clip(-999, 999)
    
    # Rename for backward compatibility ← ADD THIS
    df = df.rename(columns={
        prev_col: 'net_sales_2024',
        curr_col: 'net_sales_2025'
    })
    
    return df
```

## 12. Dashboard Header Update

```python
# In main() function, show current comparison years
prev_year, curr_year = get_year_range()
st.markdown(f"""
<div class='dashboard-header'>
    📊 Melcom KPI Dashboard - {prev_year} vs {curr_year} Performance
</div>
""", unsafe_allow_html=True)
```

This approach ensures the dashboard automatically adapts to new years while maintaining backward compatibility with existing UI code that expects column names like 'net_sales_2024' and 'net_sales_2025'.
