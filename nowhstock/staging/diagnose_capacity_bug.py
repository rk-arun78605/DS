"""
Diagnostic script to analyze capacity violation bug in staging view
Bypasses psql pager issues by using Python
"""
import psycopg2
import pandas as pd
from psycopg2.extras import RealDictCursor

# Database connection
DB_CONFIG = {
    'host': 'localhost',
    'port': 3307,
    'database': 'salesdata',
    'user': 'postgres',
    'password': 'hello'
}

def analyze_item_14722():
    """Analyze item 14722 recommendations to SPN"""
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    
    print("=" * 80)
    print("CAPACITY VIOLATION ANALYSIS: Item 14722 → SPN")
    print("=" * 80)
    
    # Get aggregates
    cursor.execute("""
        SELECT 
            COUNT(*) as source_count,
            SUM(recommended_qty) as total_recommended,
            MAX(dest_sales_used) as capacity
        FROM mv_recommendations_complete_staging
        WHERE item_code = '14722' AND dest_shop = 'SPN'
    """)
    agg = cursor.fetchone()
    print(f"\n📊 SUMMARY:")
    print(f"   Sources recommending: {agg['source_count']}")
    print(f"   Total recommended: {agg['total_recommended']}")
    print(f"   Capacity (dest_sales_used): {agg['capacity']}")
    print(f"   🚨 VIOLATION: {agg['total_recommended'] - agg['capacity']} units over capacity")
    
    # Get detailed breakdown
    cursor.execute("""
        SELECT 
            source_shop,
            source_stock,
            source_sales,
            dest_sales_used,
            recommended_qty,
            cumulative_qty
        FROM mv_recommendations_complete_staging
        WHERE item_code = '14722' AND dest_shop = 'SPN'
        ORDER BY cumulative_qty
    """)
    
    rows = cursor.fetchall()
    df = pd.DataFrame(rows)
    
    print(f"\n📋 DETAILED BREAKDOWN ({len(df)} sources):")
    print("-" * 80)
    
    # Show first 10
    print("\n🔝 First 10 sources:")
    for i, row in enumerate(df.head(10).itertuples(), 1):
        print(f"   {i}. {row.source_shop:4s} | Stock: {row.source_stock:4.0f} | Sales: {row.source_sales:4.0f} | " 
              f"Recommended: {row.recommended_qty:3.0f} | Cumulative: {row.cumulative_qty:4.0f}")
    
    # Show last 10
    print("\n🔚 Last 10 sources:")
    for i, row in enumerate(df.tail(10).itertuples(), len(df) - 9):
        print(f"   {i}. {row.source_shop:4s} | Stock: {row.source_stock:4.0f} | Sales: {row.source_sales:4.0f} | "
              f"Recommended: {row.recommended_qty:3.0f} | Cumulative: {row.cumulative_qty:4.0f}")
    
    # Find where capacity was exceeded
    capacity = agg['capacity']
    exceeded_at = df[df['cumulative_qty'] > capacity].head(1)
    
    if not exceeded_at.empty:
        idx = exceeded_at.index[0]
        print(f"\n⚠️  CAPACITY EXCEEDED AT SOURCE #{idx + 1}: {exceeded_at.iloc[0]['source_shop']}")
        print(f"   Cumulative qty: {exceeded_at.iloc[0]['cumulative_qty']}")
        print(f"   Capacity: {capacity}")
        print(f"   This source should have received 0 or reduced allocation!")
    
    # Statistics
    print(f"\n📈 STATISTICS:")
    print(f"   Mean recommended per source: {df['recommended_qty'].mean():.1f}")
    print(f"   Max single recommendation: {df['recommended_qty'].max():.0f}")
    print(f"   Sources with qty > 0: {(df['recommended_qty'] > 0).sum()}")
    
    # Save to CSV
    output_file = 'd:/Dashboard Code/NO_WH/DS/nowhstock/staging/item_14722_analysis.csv'
    df.to_csv(output_file, index=False)
    print(f"\n💾 Full details saved to: {output_file}")
    
    cursor.close()
    conn.close()
    
    return df

def check_all_violations():
    """Check all items with capacity violations"""
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    
    print("\n" + "=" * 80)
    print("SCANNING ALL ITEMS FOR CAPACITY VIOLATIONS")
    print("=" * 80)
    
    cursor.execute("""
        SELECT 
            item_code,
            dest_shop,
            COUNT(*) as source_count,
            SUM(recommended_qty) as total_recommended,
            MAX(dest_sales_used) as capacity,
            SUM(recommended_qty) - MAX(dest_sales_used) as violation_amount
        FROM mv_recommendations_complete_staging
        GROUP BY item_code, dest_shop
        HAVING SUM(recommended_qty) > MAX(dest_sales_used)
        ORDER BY SUM(recommended_qty) - MAX(dest_sales_used) DESC
        LIMIT 20
    """)
    
    violations = cursor.fetchall()
    
    if violations:
        print(f"\n🚨 FOUND {len(violations)} ITEM+DEST PAIRS WITH VIOLATIONS (showing top 20):\n")
        for i, v in enumerate(violations, 1):
            print(f"{i:3d}. Item {v['item_code']:8s} → {v['dest_shop']:4s} | "
                  f"Recommended: {v['total_recommended']:5.0f} | Capacity: {v['capacity']:5.0f} | "
                  f"Over: {v['violation_amount']:4.0f} ({v['source_count']} sources)")
    else:
        print("\n✅ NO VIOLATIONS FOUND - All recommendations respect capacity!")
    
    cursor.close()
    conn.close()
    
    return violations

if __name__ == '__main__':
    # Analyze specific item
    df = analyze_item_14722()
    
    # Check for other violations
    violations = check_all_violations()
    
    print("\n" + "=" * 80)
    print("DIAGNOSIS COMPLETE")
    print("=" * 80)
