"""
Export Production vs Test Recommendations to Excel (Optimized Version)
Creates manageable Excel file with summaries and top items only
For full data, query the database directly
"""

import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor
from datetime import datetime
import os

# Database configuration
DB_CONFIG = {
    'host': 'localhost',
    'port': 3307,
    'user': 'postgres',
    'password': 'hello',
    'database': 'salesdata'
}

def get_connection():
    """Create database connection"""
    return psycopg2.connect(**DB_CONFIG)

def get_summary_statistics(view_name):
    """Get summary statistics for a view"""
    queries = {
        'Total Recommendations': f"SELECT COUNT(*) as count FROM {view_name}",
        'Total Quantity': f"SELECT SUM(recommended_qty) as total FROM {view_name}",
        'Unique Items': f"SELECT COUNT(DISTINCT item_code) as count FROM {view_name}",
        'Unique Source Shops': f"SELECT COUNT(DISTINCT source_shop) as count FROM {view_name}",
        'Unique Dest Shops': f"SELECT COUNT(DISTINCT dest_shop) as count FROM {view_name}",
        'Priority Shop Sources': f"""
            SELECT COUNT(DISTINCT source_shop) as count 
            FROM {view_name}
            WHERE source_shop IN ('SPN', 'MSS', 'LFS', 'M03', 'KAS', 'MM1', 'MM2', 'FAR', 'KS7', 'WHL', 'MM3')
        """,
        'Same-Shop Transfers': f"""
            SELECT COUNT(*) as count 
            FROM {view_name}
            WHERE source_shop = dest_shop
        """
    }
    
    results = {}
    with get_connection() as conn:
        for stat_name, query in queries.items():
            cursor = conn.cursor()
            cursor.execute(query)
            results[stat_name] = cursor.fetchone()[0]
            cursor.close()
    
    return results

def get_shop_breakdown(view_name):
    """Get recommendations breakdown by source and destination shops"""
    query = f"""
        SELECT 
            source_shop,
            dest_shop,
            COUNT(*) as transfer_count,
            COUNT(DISTINCT item_code) as unique_items,
            SUM(recommended_qty) as total_qty,
            ROUND(AVG(recommended_qty)::numeric, 2) as avg_qty
        FROM {view_name}
        GROUP BY source_shop, dest_shop
        ORDER BY SUM(recommended_qty) DESC
    """
    
    with get_connection() as conn:
        df = pd.read_sql(query, conn)
    
    return df

def get_item_breakdown(view_name, limit=1000):
    """Get recommendations breakdown by item (top N)"""
    query = f"""
        SELECT 
            item_code,
            item_name,
            groups,
            sub_group,
            COUNT(DISTINCT source_shop) as source_shops,
            COUNT(DISTINCT dest_shop) as dest_shops,
            COUNT(*) as total_transfers,
            SUM(recommended_qty) as total_qty,
            ROUND(AVG(recommended_qty)::numeric, 2) as avg_qty
        FROM {view_name}
        GROUP BY item_code, item_name, groups, sub_group
        ORDER BY SUM(recommended_qty) DESC
        LIMIT {limit}
    """
    
    with get_connection() as conn:
        df = pd.read_sql(query, conn)
    
    return df

def get_priority_to_priority_summary():
    """Get priority-to-priority shop transfer summary"""
    query = """
        SELECT 
            source_shop,
            dest_shop,
            COUNT(*) as transfer_count,
            COUNT(DISTINCT item_code) as unique_items,
            SUM(recommended_qty) as total_qty,
            ROUND(AVG(recommended_qty)::numeric, 2) as avg_qty,
            STRING_AGG(DISTINCT groups, ', ') as product_groups
        FROM mv_recommendations_complete_test
        WHERE source_shop IN ('SPN', 'MSS', 'LFS', 'M03', 'KAS', 'MM1', 'MM2', 'FAR', 'KS7', 'WHL', 'MM3')
        GROUP BY source_shop, dest_shop
        ORDER BY SUM(recommended_qty) DESC
    """
    
    with get_connection() as conn:
        df = pd.read_sql(query, conn)
    
    return df

def get_priority_to_priority_top_items(limit=500):
    """Get top items for priority-to-priority transfers"""
    query = f"""
        SELECT 
            source_shop,
            dest_shop,
            item_code,
            item_name,
            groups,
            sub_group,
            recommended_qty,
            source_stock,
            source_sales,
            dest_sales,
            source_grn_age,
            source_expiry_days,
            expiry_status,
            priority_rank
        FROM mv_recommendations_complete_test
        WHERE source_shop IN ('SPN', 'MSS', 'LFS', 'M03', 'KAS', 'MM1', 'MM2', 'FAR', 'KS7', 'WHL', 'MM3')
        ORDER BY recommended_qty DESC
        LIMIT {limit}
    """
    
    with get_connection() as conn:
        df = pd.read_sql(query, conn)
    
    return df

def get_new_vs_removed_summary():
    """Get summary of new and removed recommendations"""
    query = """
        WITH prod_keys AS (
            SELECT source_shop, dest_shop, item_code FROM mv_recommendations_complete
        ),
        test_keys AS (
            SELECT source_shop, dest_shop, item_code FROM mv_recommendations_complete_test
        )
        SELECT
            'NEW (in test only)' as category,
            COUNT(*) as count,
            COUNT(DISTINCT t.item_code) as unique_items,
            COUNT(DISTINCT t.source_shop) as unique_sources
        FROM test_keys t
        WHERE NOT EXISTS (
            SELECT 1 FROM prod_keys p 
            WHERE p.source_shop = t.source_shop 
            AND p.dest_shop = t.dest_shop 
            AND p.item_code = t.item_code
        )
        UNION ALL
        SELECT
            'REMOVED (in prod only)' as category,
            COUNT(*) as count,
            COUNT(DISTINCT p.item_code) as unique_items,
            COUNT(DISTINCT p.source_shop) as unique_sources
        FROM prod_keys p
        WHERE NOT EXISTS (
            SELECT 1 FROM test_keys t
            WHERE t.source_shop = p.source_shop 
            AND t.dest_shop = p.dest_shop 
            AND t.item_code = p.item_code
        )
    """
    
    with get_connection() as conn:
        df = pd.read_sql(query, conn)
    
    return df

def get_source_shop_comparison():
    """Compare source shops between production and test"""
    query = """
        SELECT 
            COALESCE(p.source_shop, t.source_shop) as source_shop,
            COALESCE(p.prod_count, 0) as prod_recommendations,
            COALESCE(t.test_count, 0) as test_recommendations,
            COALESCE(t.test_count, 0) - COALESCE(p.prod_count, 0) as difference,
            CASE 
                WHEN p.source_shop IN ('SPN', 'MSS', 'LFS', 'M03', 'KAS', 'MM1', 'MM2', 'FAR', 'KS7', 'WHL', 'MM3')
                THEN 'Priority Shop (NEW)'
                ELSE 'Regular Shop'
            END as shop_type
        FROM (
            SELECT source_shop, COUNT(*) as prod_count
            FROM mv_recommendations_complete
            GROUP BY source_shop
        ) p
        FULL OUTER JOIN (
            SELECT source_shop, COUNT(*) as test_count
            FROM mv_recommendations_complete_test
            GROUP BY source_shop
        ) t ON p.source_shop = t.source_shop
        ORDER BY COALESCE(t.test_count, 0) - COALESCE(p.prod_count, 0) DESC
    """
    
    with get_connection() as conn:
        df = pd.read_sql(query, conn)
    
    return df

def create_comparison_summary(prod_stats, test_stats):
    """Create side-by-side comparison summary"""
    comparison = []
    
    for metric in prod_stats.keys():
        prod_val = prod_stats[metric]
        test_val = test_stats[metric]
        
        if prod_val > 0:
            diff = test_val - prod_val
            pct_change = (diff / prod_val) * 100
        else:
            diff = test_val
            pct_change = 100.0 if test_val > 0 else 0
        
        comparison.append({
            'Metric': metric,
            'Production': prod_val,
            'Test': test_val,
            'Difference': diff,
            'Change %': round(pct_change, 2)
        })
    
    return pd.DataFrame(comparison)

def export_to_excel():
    """Main function to export summary data to Excel"""
    print("=" * 80)
    print("EXPORTING RECOMMENDATIONS COMPARISON (SUMMARY VERSION)")
    print("=" * 80)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    print("NOTE: Exporting SUMMARIES only (full data = 1.3M rows, too large for Excel)")
    print("      For full data, query mv_recommendations_complete_test directly")
    print()
    
    # Output filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f"Recommendations_Comparison_Summary_{timestamp}.xlsx"
    output_path = os.path.join(os.path.dirname(__file__), output_file)
    
    # Create Excel writer
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        
        # Sheet 1: Summary Statistics Comparison
        print("📊 Fetching summary statistics...")
        prod_stats = get_summary_statistics('mv_recommendations_complete')
        test_stats = get_summary_statistics('mv_recommendations_complete_test')
        comparison_df = create_comparison_summary(prod_stats, test_stats)
        comparison_df.to_excel(writer, sheet_name='1. Summary', index=False)
        print(f"   ✅ Summary: {len(comparison_df)} metrics compared")
        print(f"      Production: {prod_stats['Total Recommendations']:,} recommendations")
        print(f"      Test: {test_stats['Total Recommendations']:,} recommendations")
        print(f"      Increase: +{test_stats['Total Recommendations'] - prod_stats['Total Recommendations']:,} ({((test_stats['Total Recommendations'] - prod_stats['Total Recommendations']) / prod_stats['Total Recommendations'] * 100):.1f}%)")
        
        # Sheet 2: Source Shop Comparison
        print("🏪 Comparing source shops...")
        source_comp = get_source_shop_comparison()
        source_comp.to_excel(writer, sheet_name='2. Source Shop Comparison', index=False)
        priority_sources = source_comp[source_comp['shop_type'].str.contains('Priority', na=False)]
        print(f"   ✅ Source Shops: {len(source_comp)} shops analyzed")
        print(f"      Priority shops as sources: {len(priority_sources)} (NEW)")
        
        # Sheet 3: Priority-to-Priority Summary
        print("🔄 Analyzing priority-to-priority transfers...")
        priority_summary = get_priority_to_priority_summary()
        if len(priority_summary) > 0:
            priority_summary.to_excel(writer, sheet_name='3. Priority-to-Priority', index=False)
            print(f"   ✅ Priority-to-Priority: {len(priority_summary)} shop pairs")
            print(f"      Total items: {priority_summary['unique_items'].sum():,}")
            print(f"      Total quantity: {priority_summary['total_qty'].sum():,.0f}")
        else:
            empty_df = pd.DataFrame({'Note': ['No priority-to-priority transfers found']})
            empty_df.to_excel(writer, sheet_name='3. Priority-to-Priority', index=False)
            print(f"   ⚠️  No priority-to-priority transfers")
        
        # Sheet 4: Priority-to-Priority Top 500 Items
        if len(priority_summary) > 0:
            print("📦 Fetching top priority-to-priority items...")
            priority_items = get_priority_to_priority_top_items(500)
            priority_items.to_excel(writer, sheet_name='4. Priority Items (Top 500)', index=False)
            print(f"   ✅ Top 500 priority-to-priority items exported")
        
        # Sheet 5: Production Shop Breakdown
        print("🏪 Analyzing production shop pairs...")
        prod_shop_breakdown = get_shop_breakdown('mv_recommendations_complete')
        prod_shop_breakdown.to_excel(writer, sheet_name='5. Prod Shop Breakdown', index=False)
        print(f"   ✅ Production: {len(prod_shop_breakdown)} shop pairs")
        
        # Sheet 6: Test Shop Breakdown
        print("🏪 Analyzing test shop pairs...")
        test_shop_breakdown = get_shop_breakdown('mv_recommendations_complete_test')
        test_shop_breakdown.to_excel(writer, sheet_name='6. Test Shop Breakdown', index=False)
        print(f"   ✅ Test: {len(test_shop_breakdown)} shop pairs")
        
        # Sheet 7: Production Top Items
        print("📦 Fetching top production items...")
        prod_items = get_item_breakdown('mv_recommendations_complete', 1000)
        prod_items.to_excel(writer, sheet_name='7. Prod Items (Top 1000)', index=False)
        print(f"   ✅ Production: Top {len(prod_items)} items")
        
        # Sheet 8: Test Top Items
        print("📦 Fetching top test items...")
        test_items = get_item_breakdown('mv_recommendations_complete_test', 1000)
        test_items.to_excel(writer, sheet_name='8. Test Items (Top 1000)', index=False)
        print(f"   ✅ Test: Top {len(test_items)} items")
        
        # Sheet 9: New vs Removed Summary
        print("📊 Analyzing new vs removed recommendations...")
        new_removed = get_new_vs_removed_summary()
        new_removed.to_excel(writer, sheet_name='9. New vs Removed Summary', index=False)
        print(f"   ✅ New vs Removed summary created")
    
    print()
    print("=" * 80)
    print("✅ EXPORT COMPLETE")
    print("=" * 80)
    print(f"📁 File saved: {output_path}")
    print(f"📊 File size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")
    print()
    print("📋 SHEETS CREATED:")
    print("   1. Summary - Key metrics comparison")
    print("   2. Source Shop Comparison - All shops (highlights priority shops)")
    print("   3. Priority-to-Priority - Summary by shop pair")
    print("   4. Priority Items (Top 500) - Detailed top priority-to-priority items")
    print("   5. Prod Shop Breakdown - Production shop pair summary")
    print("   6. Test Shop Breakdown - Test shop pair summary")
    print("   7. Prod Items (Top 1000) - Top production items")
    print("   8. Test Items (Top 1000) - Top test items")
    print("   9. New vs Removed Summary - High-level changes")
    print()
    print("🔍 KEY FINDINGS:")
    print(f"   • Production has {prod_stats['Total Recommendations']:,} recommendations")
    print(f"   • Test has {test_stats['Total Recommendations']:,} recommendations")
    print(f"   • Increase: +{test_stats['Total Recommendations'] - prod_stats['Total Recommendations']:,} (+{((test_stats['Total Recommendations'] - prod_stats['Total Recommendations']) / prod_stats['Total Recommendations'] * 100):.1f}%)")
    print(f"   • Priority shop sources: {test_stats['Priority Shop Sources']} (was 0 in production)")
    print(f"   • Same-shop transfers: {test_stats['Same-Shop Transfers']} (should be 0) ✅" if test_stats['Same-Shop Transfers'] == 0 else f"   • Same-shop transfers: {test_stats['Same-Shop Transfers']} ❌ ERROR!")
    print()
    print("💡 NEXT STEPS:")
    print("   1. Review 'Summary' sheet - validate expected increases")
    print("   2. Check 'Source Shop Comparison' - verify priority shops as sources")
    print("   3. Review 'Priority-to-Priority' - validate new transfers")
    print("   4. Examine 'Priority Items (Top 500)' - spot check recommendations")
    print("   5. Compare shop breakdowns - ensure logic is consistent")
    print()
    print("📌 NOTE: For detailed item-level comparison, query database directly:")
    print("   SELECT * FROM mv_recommendations_complete_test WHERE source_shop = 'SPN';")
    print("=" * 80)

if __name__ == "__main__":
    try:
        export_to_excel()
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        input("\nPress Enter to exit...")
