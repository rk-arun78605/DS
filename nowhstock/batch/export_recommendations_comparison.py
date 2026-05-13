"""
Export Production vs Test Recommendations to Excel for Validation
Creates side-by-side comparison with multiple sheets for analysis
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

def fetch_recommendations(view_name):
    """Fetch all recommendations from specified view"""
    query = f"""
        SELECT 
            item_code,
            item_name,
            groups,
            sub_group,
            item_type,
            supplier_name,
            source_shop,
            source_stock,
            source_sales,
            source_last_grn,
            source_grn_age,
            source_wh_grn_date,
            source_expiry_date,
            source_expiry_days,
            expiry_check,
            expiry_status,
            dest_shop,
            dest_stock,
            dest_sales,
            dest_last_grn,
            dest_grn_age,
            dest_wh_grn_date,
            dest_wh_grn_plus_30,
            dest_wh_grn_30d_sales,
            dest_sales_used,
            priority_rank,
            recommended_qty,
            cumulative_qty,
            dest_remaining_cap_before,
            dest_updated_stock,
            dest_final_stock_days,
            remark
        FROM {view_name}
        ORDER BY 
            dest_shop,
            priority_rank,
            item_code,
            source_grn_age DESC
    """
    
    with get_connection() as conn:
        df = pd.read_sql(query, conn)
        # Remove timezone from datetime columns to prevent Excel corruption
        for col in df.select_dtypes(include=['datetime64[ns, UTC]']).columns:
            df[col] = df[col].dt.tz_localize(None)
    
    return df

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
            ROUND(AVG(recommended_qty), 2) as avg_qty,
            MIN(recommended_qty) as min_qty,
            MAX(recommended_qty) as max_qty
        FROM {view_name}
        GROUP BY source_shop, dest_shop
        ORDER BY SUM(recommended_qty) DESC
    """
    
    with get_connection() as conn:
        df = pd.read_sql(query, conn)
    
    return df

def get_item_breakdown(view_name):
    """Get recommendations breakdown by item"""
    query = f"""
        SELECT 
            item_code,
            item_name,
            COUNT(DISTINCT source_shop) as source_shops,
            COUNT(DISTINCT dest_shop) as dest_shops,
            COUNT(*) as total_transfers,
            SUM(recommended_qty) as total_qty,
            ROUND(AVG(recommended_qty), 2) as avg_qty,
            MIN(recommended_qty) as min_qty,
            MAX(recommended_qty) as max_qty
        FROM {view_name}
        GROUP BY item_code, item_name
        ORDER BY SUM(recommended_qty) DESC
        LIMIT 500
    """
    
    with get_connection() as conn:
        df = pd.read_sql(query, conn)
    
    return df

def get_priority_to_priority_transfers(view_name):
    """Get priority-to-priority shop transfers (NEW in test view)"""
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
            dest_sales_used,
            source_grn_age,
            source_expiry_days,
            expiry_status,
            priority_rank
        FROM {view_name}
        WHERE source_shop IN ('SPN', 'MSS', 'LFS', 'M03', 'KAS', 'MM1', 'MM2', 'FAR', 'KS7', 'WHL', 'MM3')
        ORDER BY 
            source_shop,
            dest_shop,
            recommended_qty DESC
    """
    
    with get_connection() as conn:
        df = pd.read_sql(query, conn)
        # Remove timezone from datetime columns
        for col in df.select_dtypes(include=['datetime64[ns, UTC]']).columns:
            df[col] = df[col].dt.tz_localize(None)
    
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
    """Main function to export all data to Excel"""
    print("=" * 80)
    print("EXPORTING RECOMMENDATIONS FOR VALIDATION")
    print("=" * 80)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Output filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f"Recommendations_Comparison_{timestamp}.xlsx"
    output_path = os.path.join(os.path.dirname(__file__), output_file)
    
    # Create Excel writer
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        
        # Sheet 1: Summary Statistics Comparison
        print("📊 Fetching summary statistics...")
        prod_stats = get_summary_statistics('mv_recommendations_complete')
        test_stats = get_summary_statistics('mv_recommendations_complete_test')
        comparison_df = create_comparison_summary(prod_stats, test_stats)
        comparison_df.to_excel(writer, sheet_name='1. Summary Comparison', index=False)
        print(f"   ✅ Summary: {len(comparison_df)} metrics compared")
        
        # Sheet 2: Production Recommendations (ALL)
        print("📥 Fetching PRODUCTION recommendations...")
        prod_df = fetch_recommendations('mv_recommendations_complete')
        prod_df.to_excel(writer, sheet_name='2. Production (ALL)', index=False)
        print(f"   ✅ Production: {len(prod_df):,} recommendations exported")
        
        # Sheet 3: Test Recommendations (ALL)
        print("📥 Fetching TEST recommendations...")
        test_df = fetch_recommendations('mv_recommendations_complete_test')
        test_df.to_excel(writer, sheet_name='3. Test (ALL)', index=False)
        print(f"   ✅ Test: {len(test_df):,} recommendations exported")
        
        # Sheet 4: Priority-to-Priority Transfers (NEW in test)
        print("🔄 Fetching priority-to-priority transfers...")
        priority_df = get_priority_to_priority_transfers('mv_recommendations_complete_test')
        if len(priority_df) > 0:
            priority_df.to_excel(writer, sheet_name='4. Priority-to-Priority', index=False)
            print(f"   ✅ Priority-to-Priority: {len(priority_df):,} transfers (NEW)")
        else:
            empty_df = pd.DataFrame({'Note': ['No priority-to-priority transfers found. Priority shops may not have excess stock.']})
            empty_df.to_excel(writer, sheet_name='4. Priority-to-Priority', index=False)
            print(f"   ⚠️  Priority-to-Priority: 0 transfers (priority shops may not have excess stock)")
        
        # Sheet 5: Production Shop Breakdown
        print("🏪 Analyzing shop breakdowns...")
        prod_shop_breakdown = get_shop_breakdown('mv_recommendations_complete')
        prod_shop_breakdown.to_excel(writer, sheet_name='5. Prod Shop Breakdown', index=False)
        print(f"   ✅ Production: {len(prod_shop_breakdown)} shop pairs")
        
        # Sheet 6: Test Shop Breakdown
        test_shop_breakdown = get_shop_breakdown('mv_recommendations_complete_test')
        test_shop_breakdown.to_excel(writer, sheet_name='6. Test Shop Breakdown', index=False)
        print(f"   ✅ Test: {len(test_shop_breakdown)} shop pairs")
        
        # Sheet 7: Production Item Breakdown (Top 500)
        print("📦 Analyzing item breakdowns...")
        prod_item_breakdown = get_item_breakdown('mv_recommendations_complete')
        prod_item_breakdown.to_excel(writer, sheet_name='7. Prod Item Breakdown', index=False)
        print(f"   ✅ Production: Top {len(prod_item_breakdown)} items")
        
        # Sheet 8: Test Item Breakdown (Top 500)
        test_item_breakdown = get_item_breakdown('mv_recommendations_complete_test')
        test_item_breakdown.to_excel(writer, sheet_name='8. Test Item Breakdown', index=False)
        print(f"   ✅ Test: Top {len(test_item_breakdown)} items")
        
        # Sheet 9: NEW Recommendations (in test but not in production)
        print("🆕 Finding NEW recommendations...")
        prod_keys = prod_df[['source_shop', 'dest_shop', 'item_code']].apply(
            lambda x: f"{x['source_shop']}|{x['dest_shop']}|{x['item_code']}", axis=1
        )
        test_keys = test_df[['source_shop', 'dest_shop', 'item_code']].apply(
            lambda x: f"{x['source_shop']}|{x['dest_shop']}|{x['item_code']}", axis=1
        )
        
        new_mask = ~test_keys.isin(prod_keys)
        new_recs = test_df[new_mask].copy()
        new_recs['Note'] = 'NEW in test view (not in production)'
        new_recs.to_excel(writer, sheet_name='9. NEW Recommendations', index=False)
        print(f"   ✅ NEW: {len(new_recs):,} recommendations added in test view")
        
        # Sheet 10: REMOVED Recommendations (in production but not in test)
        print("❌ Finding REMOVED recommendations...")
        removed_mask = ~prod_keys.isin(test_keys)
        removed_recs = prod_df[removed_mask].copy()
        removed_recs['Note'] = 'REMOVED in test view (was in production)'
        removed_recs.to_excel(writer, sheet_name='10. REMOVED Recommendations', index=False)
        print(f"   ✅ REMOVED: {len(removed_recs):,} recommendations removed in test view")
        
        # Sheet 11: Quantity Changes (same source-dest-item, different qty)
        print("📊 Finding quantity changes...")
        prod_dict = prod_df.set_index(['source_shop', 'dest_shop', 'item_code'])['recommended_qty'].to_dict()
        test_dict = test_df.set_index(['source_shop', 'dest_shop', 'item_code'])['recommended_qty'].to_dict()
        
        qty_changes = []
        for key in set(prod_dict.keys()).intersection(test_dict.keys()):
            prod_qty = prod_dict[key]
            test_qty = test_dict[key]
            if prod_qty != test_qty:
                qty_changes.append({
                    'source_shop': key[0],
                    'dest_shop': key[1],
                    'item_code': key[2],
                    'production_qty': prod_qty,
                    'test_qty': test_qty,
                    'difference': test_qty - prod_qty,
                    'change_pct': round((test_qty - prod_qty) / prod_qty * 100, 2) if prod_qty > 0 else 0
                })
        
        if qty_changes:
            qty_changes_df = pd.DataFrame(qty_changes).sort_values('difference', ascending=False)
            qty_changes_df.to_excel(writer, sheet_name='11. Quantity Changes', index=False)
            print(f"   ✅ Quantity Changes: {len(qty_changes_df):,} recommendations with different quantities")
        else:
            empty_df = pd.DataFrame({'Note': ['No quantity changes found']})
            empty_df.to_excel(writer, sheet_name='11. Quantity Changes', index=False)
            print(f"   ✅ Quantity Changes: 0 (all matching recommendations have same quantities)")
    
    print()
    print("=" * 80)
    print("✅ EXPORT COMPLETE")
    print("=" * 80)
    print(f"📁 File saved: {output_path}")
    print(f"📊 File size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")
    print()
    print("📋 SHEETS CREATED:")
    print("   1. Summary Comparison - Key metrics side-by-side")
    print("   2. Production (ALL) - All production recommendations")
    print("   3. Test (ALL) - All test recommendations")
    print("   4. Priority-to-Priority - NEW transfers between priority shops")
    print("   5. Prod Shop Breakdown - Production shop pair analysis")
    print("   6. Test Shop Breakdown - Test shop pair analysis")
    print("   7. Prod Item Breakdown - Top 500 items in production")
    print("   8. Test Item Breakdown - Top 500 items in test")
    print("   9. NEW Recommendations - Only in test view")
    print("   10. REMOVED Recommendations - Only in production view")
    print("   11. Quantity Changes - Same transfers, different quantities")
    print()
    print("🔍 VALIDATION STEPS:")
    print("   1. Check 'Summary Comparison' sheet - verify expected increases")
    print("   2. Review 'Priority-to-Priority' sheet - validate new logic")
    print("   3. Compare 'NEW Recommendations' - should be mostly priority-to-priority")
    print("   4. Check 'REMOVED Recommendations' - should be minimal")
    print("   5. Review 'Quantity Changes' - verify allocation logic changes")
    print()
    print("💡 TIP: Use Excel filters and pivot tables for deeper analysis")
    print("=" * 80)

if __name__ == "__main__":
    try:
        export_to_excel()
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        input("\nPress Enter to exit...")
