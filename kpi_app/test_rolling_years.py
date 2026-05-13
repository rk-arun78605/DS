"""
Verify Rolling 5-Year MTD/YTD Data
Quick script to check if the new rolling functions work correctly
"""
import psycopg2
from datetime import datetime, timedelta
from contextlib import contextmanager

@contextmanager
def get_db_connection():
    conn = psycopg2.connect(
        host='localhost',
        port=3307,
        user='postgres',
        password='hello',
        database='salesdata'
    )
    try:
        yield conn
    finally:
        conn.close()

def get_available_years():
    """Get available sales tables"""
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT tablename 
                FROM pg_tables 
                WHERE schemaname = 'public' 
                  AND tablename LIKE 'sales_%'
                  AND tablename ~ '^sales_[0-9]{4}$'
                ORDER BY tablename
            """)
            tables = cur.fetchall()
            years = sorted([int(table[0].split('_')[1]) for table in tables])
            return years

def get_rolling_5_years():
    """Get last 5 years <= current year"""
    current_year = datetime.now().year
    available_years = get_available_years()
    available_years = [y for y in available_years if y <= current_year]
    return available_years[-5:] if len(available_years) >= 5 else available_years

def get_mtd_dates():
    """Get MTD date range"""
    today = datetime.today()
    yesterday = today - timedelta(days=1)
    start_date = yesterday.replace(day=1)
    return start_date.date(), yesterday.date()

def test_rolling_mtd():
    """Test rolling MTD data"""
    print("=" * 80)
    print("ROLLING 5-YEAR MTD DATA TEST")
    print("=" * 80)
    
    rolling_years = get_rolling_5_years()
    print(f"\n✅ Rolling years: {rolling_years}")
    
    start_date, end_date = get_mtd_dates()
    print(f"✅ MTD date range: {start_date} to {end_date}")
    print(f"   Days in range: {(end_date - start_date).days + 1}")
    
    print("\n" + "=" * 80)
    print("MTD SALES DATA BY YEAR")
    print("=" * 80)
    
    with get_db_connection() as conn:
        for year in rolling_years:
            # Determine column case
            if year <= 2025:
                qty_col = '"QTY"'
                sales_col = '"NET_SALES"'
                date_col = '"DATE_INVOICE"'
            else:
                qty_col = 'qty'
                sales_col = 'net_sales'
                date_col = 'date_invoice'
            
            query = f"""
                SELECT 
                    COUNT(*) as record_count,
                    MIN({date_col}::date) as first_date,
                    MAX({date_col}::date) as last_date,
                    SUM({qty_col}) as total_qty,
                    SUM({sales_col}) as total_sales
                FROM sales_{year}
                WHERE {date_col}::date >= %s 
                  AND {date_col}::date <= %s
            """
            
            with conn.cursor() as cur:
                cur.execute(query, (
                    start_date.replace(year=year),
                    end_date.replace(year=year)
                ))
                result = cur.fetchone()
                
                record_count, first_date, last_date, total_qty, total_sales = result
                
                print(f"\n{year}:")
                print(f"  Records: {record_count:,}")
                print(f"  Date range: {first_date} to {last_date}")
                print(f"  Total Qty: {int(total_qty) if total_qty else 0:,}")
                print(f"  Total Sales: GH₵ {total_sales:,.2f}" if total_sales else "  Total Sales: GH₵ 0.00")
                
                # Calculate YoY growth
                if len(rolling_years) > 1 and year != rolling_years[0]:
                    prev_year = rolling_years[rolling_years.index(year) - 1]
                    
                    prev_query = f"""
                        SELECT SUM({sales_col if year <= 2025 else sales_col})
                        FROM sales_{prev_year}
                        WHERE {date_col}::date >= %s 
                          AND {date_col}::date <= %s
                    """
                    
                    cur.execute(prev_query, (
                        start_date.replace(year=prev_year),
                        end_date.replace(year=prev_year)
                    ))
                    prev_sales = cur.fetchone()[0] or 0
                    
                    if prev_sales > 0:
                        growth = ((total_sales or 0) - prev_sales) / prev_sales * 100
                        arrow = "↑" if growth >= 0 else "↓"
                        color = "green" if growth >= 0 else "red"
                        print(f"  YoY Growth: {arrow} {abs(growth):.1f}% ({color})")
    
    print("\n" + "=" * 80)
    print("✅ TEST COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    try:
        test_rolling_mtd()
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
