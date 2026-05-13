import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="Serial Number Tracking - Anomaly Detection",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main > div {padding-top: 2rem;}
    .stMetric {background-color: #f0f2f6; padding: 15px; border-radius: 5px;}
    .anomaly-high {background-color: #ffebee; padding: 10px; border-radius: 5px; border-left: 4px solid #d32f2f;}
    .anomaly-medium {background-color: #fff3e0; padding: 10px; border-radius: 5px; border-left: 4px solid #f57c00;}
    h1 {color: #d32f2f;}
    h2 {color: #f57c00;}
    </style>
""", unsafe_allow_html=True)

# Data loading
@st.cache_data(ttl=3600)
def load_data():
    """Load serial number data from Excel file"""
    file_path = r"d:\Dashboard Code\NO_WH\DS\SerialNoReport\file from zaheed\Serial no report nov and dec.xlsx"
    df = pd.read_excel(file_path)
    
    # Convert date columns
    date_columns = ['DT_DOC_DATE', 'LOADED_DATETIME', 'LOADINGDATE', 'DT_LOAD_DATE', 
                    'DT_MOD_DATE', 'DT_INVOICE_DATE']
    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    
    return df

@st.cache_data(ttl=3600)
def prepare_tracking_features(df):
    """Prepare features for anomaly tracking"""
    track_df = df.copy()
    
    # Calculate loading time (from document date to loading datetime)
    track_df['loading_time_hours'] = (track_df['LOADED_DATETIME'] - track_df['DT_DOC_DATE']).dt.total_seconds() / 3600
    
    # Calculate transit time (from loading to shop received)
    track_df['transit_time_hours'] = (track_df['DT_MOD_DATE'] - track_df['LOADED_DATETIME']).dt.total_seconds() / 3600
    
    # Calculate time to sale after receipt
    track_df['time_to_sale_days'] = (track_df['DT_INVOICE_DATE'] - track_df['DT_MOD_DATE']).dt.days
    
    # Shop mismatch detection
    track_df['shop_mismatch'] = (track_df['VC_SHOP_CODE'] != track_df['SHOP_SOLD']) & (~track_df['SHOP_SOLD'].isna())
    
    # Serial number mismatch detection
    track_df['serial_mismatch'] = (track_df['SERIAL_NO'] != track_df['SHOP_SERAIL_NO']) & (~track_df['SHOP_SERAIL_NO'].isna())
    
    # Vehicle change detection
    track_df['vehicle_changed'] = (track_df['VC_VEHICLE_NO'] != track_df['VC_VEHICLE_NO.1']) & (~track_df['VC_VEHICLE_NO.1'].isna())
    
    # Flag items with missing key data
    track_df['has_complete_data'] = ~(track_df['LOADED_DATETIME'].isna() | track_df['VC_VEHICLE_NO'].isna())
    
    # Days in transit
    track_df['days_in_transit'] = track_df['transit_time_hours'] / 24
    
    # Extract time features
    track_df['load_hour'] = track_df['LOADED_DATETIME'].dt.hour
    track_df['load_day_of_week'] = track_df['LOADED_DATETIME'].dt.dayofweek
    track_df['load_month'] = track_df['LOADED_DATETIME'].dt.month
    
    return track_df

def detect_loading_time_anomalies(df):
    """Detect items with abnormal loading times"""
    loading_df = df[df['has_complete_data'] & (df['loading_time_hours'] >= 0) & (df['loading_time_hours'] < 10000)].copy()
    
    if len(loading_df) < 10:
        return None
    
    # Use Isolation Forest on loading time features
    features = loading_df[['loading_time_hours', 'load_hour', 'load_day_of_week']].fillna(0)
    
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    iso_forest = IsolationForest(contamination=0.15, random_state=42)
    loading_df['anomaly'] = iso_forest.fit_predict(features_scaled)
    loading_df['anomaly_score'] = iso_forest.score_samples(features_scaled)
    
    anomalies = loading_df[loading_df['anomaly'] == -1].copy()
    anomalies['anomaly_severity'] = pd.cut(anomalies['loading_time_hours'], 
                                            bins=[0, 24, 72, 168, float('inf')],
                                            labels=['Normal', 'Delayed', 'Critical', 'Severe'])
    
    return anomalies

def detect_transit_anomalies(df):
    """Detect items with abnormal transit times"""
    transit_df = df[(df['transit_time_hours'] >= 0) & (df['transit_time_hours'] < 10000)].copy()
    
    if len(transit_df) < 10:
        return None
    
    features = transit_df[['transit_time_hours', 'days_in_transit']].fillna(0)
    
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    iso_forest = IsolationForest(contamination=0.15, random_state=42)
    transit_df['anomaly'] = iso_forest.fit_predict(features_scaled)
    transit_df['anomaly_score'] = iso_forest.score_samples(features_scaled)
    
    return transit_df[transit_df['anomaly'] == -1]

def find_shop_mismatches(df):
    """Find items ordered for one shop but sold in another"""
    mismatches = df[df['shop_mismatch'] == True].copy()
    
    if len(mismatches) == 0:
        return None, None
    
    # Calculate frequency of mismatch patterns
    mismatch_patterns = mismatches.groupby(['VC_SHOP_CODE', 'SHOP_SOLD']).agg({
        'SERIAL_NO': 'count',
        'NU_SELLING_PRICE': 'sum'
    }).reset_index()
    mismatch_patterns.columns = ['Ordered_Shop', 'Sold_Shop', 'Count', 'Total_Value']
    mismatch_patterns = mismatch_patterns.sort_values('Count', ascending=False)
    
    return mismatches, mismatch_patterns

# Main app
def main():
    st.title("🚨 Serial Number Tracking - Anomaly Detection")
    st.markdown("**Track loading anomalies, transit issues, and shop mismatches**")
    
    # Load data
    with st.spinner("Loading data..."):
        df = load_data()
        track_df = prepare_tracking_features(df)
    
    # Sidebar filters
    st.sidebar.header("🔍 Filters")
    
    warehouses = ['All'] + sorted(df['VC_WH_CODE'].dropna().unique().tolist())
    selected_warehouse = st.sidebar.selectbox("Warehouse", warehouses)
    
    shops = ['All'] + sorted(df['VC_SHOP_CODE'].dropna().unique().tolist())
    selected_shop = st.sidebar.selectbox("Shop", shops)
    
    date_range = st.sidebar.date_input(
        "Date Range",
        value=(df['LOADED_DATETIME'].min(), df['LOADED_DATETIME'].max()),
        min_value=df['LOADED_DATETIME'].min(),
        max_value=datetime.today()
    )
    
    # Apply filters
    filtered_df = track_df.copy()
    if selected_warehouse != 'All':
        filtered_df = filtered_df[filtered_df['VC_WH_CODE'] == selected_warehouse]
    if selected_shop != 'All':
        filtered_df = filtered_df[filtered_df['VC_SHOP_CODE'] == selected_shop]
    if len(date_range) == 2:
        filtered_df = filtered_df[
            (filtered_df['LOADED_DATETIME'].dt.date >= date_range[0]) & 
            (filtered_df['LOADED_DATETIME'].dt.date <= date_range[1])
        ]
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Overview", 
        "🚛 Loading Time Anomalies", 
        "🚨 Shop Mismatches",
        "📦 Transit Issues"
    ])
    
    # TAB 1: Overview
    with tab1:
        st.header("Key Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        total_items = len(filtered_df)
        shop_mismatches = filtered_df['shop_mismatch'].sum()
        serial_mismatches = filtered_df['serial_mismatch'].sum()
        avg_loading_time = filtered_df[filtered_df['loading_time_hours'] >= 0]['loading_time_hours'].mean()
        
        col1.metric("Total Items", f"{total_items:,}")
        col2.metric("Shop Mismatches", f"{shop_mismatches:,}", 
                   f"{shop_mismatches/total_items*100:.1f}%" if total_items > 0 else "0%")
        col3.metric("Serial Number Mismatches", f"{serial_mismatches:,}")
        col4.metric("Avg Loading Time", f"{avg_loading_time:.1f}h" if not pd.isna(avg_loading_time) else "N/A")
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Loading Time Distribution")
            loading_dist = filtered_df[
                (filtered_df['loading_time_hours'] >= 0) & 
                (filtered_df['loading_time_hours'] < 200)
            ].copy()
            if len(loading_dist) > 0:
                fig = px.histogram(loading_dist, x='loading_time_hours', nbins=50,
                                 labels={'loading_time_hours': 'Loading Time (hours)'})
                fig.add_vline(x=24, line_dash="dash", line_color="red", 
                             annotation_text="24 hours")
                fig.add_vline(x=72, line_dash="dash", line_color="orange", 
                             annotation_text="72 hours")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No loading time data available")
        
        with col2:
            st.subheader("Shop Mismatch Overview")
            mismatch_summary = pd.DataFrame({
                'Type': ['Shop Matches', 'Shop Mismatches'],
                'Count': [total_items - shop_mismatches, shop_mismatches]
            })
            fig = px.pie(mismatch_summary, values='Count', names='Type',
                        color='Type', color_discrete_map={
                            'Shop Matches': '#2ca02c', 
                            'Shop Mismatches': '#d62728'
                        })
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Top Warehouses by Volume")
            wh_volume = filtered_df.groupby('VC_WAREHOUSE_DESC').size().sort_values(ascending=False).head(10)
            fig = px.bar(x=wh_volume.values, y=wh_volume.index, orientation='h',
                        labels={'x': 'Items Loaded', 'y': 'Warehouse'})
            fig.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Items Loaded Over Time")
            loading_by_date = filtered_df.groupby(
                filtered_df['LOADED_DATETIME'].dt.date
            ).size().reset_index()
            loading_by_date.columns = ['Date', 'Count']
            fig = px.line(loading_by_date, x='Date', y='Count', markers=True)
            st.plotly_chart(fig, use_container_width=True)
    
    # TAB 2: Loading Time Anomalies
    with tab2:
        st.header("🚛 Loading Time Anomalies Detection")
        st.markdown("**Items with unusual loading times from warehouse document creation to vehicle loading**")
        
        with st.spinner("Detecting loading time anomalies..."):
            anomalies = detect_loading_time_anomalies(filtered_df)
        
        if anomalies is not None and len(anomalies) > 0:
            st.error(f"⚠️ Found {len(anomalies)} items with anomalous loading times")
            
            col1, col2, col3, col4 = st.columns(4)
            
            severity_counts = anomalies['anomaly_severity'].value_counts()
            col1.metric("Normal Delays", severity_counts.get('Normal', 0))
            col2.metric("Delayed", severity_counts.get('Delayed', 0))
            col3.metric("Critical", severity_counts.get('Critical', 0))
            col4.metric("Severe", severity_counts.get('Severe', 0))
            
            st.markdown("---")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Loading Time vs Anomaly Score")
                fig = px.scatter(anomalies, x='loading_time_hours', y='anomaly_score',
                               color='anomaly_severity',
                               hover_data=['VC_ITEM_DESC', 'VC_WH_CODE', 'VC_VEHICLE_NO'],
                               color_discrete_map={
                                   'Normal': '#2ca02c',
                                   'Delayed': '#ff7f0e',
                                   'Critical': '#d62728',
                                   'Severe': '#8b0000'
                               })
                fig.update_layout(xaxis_title="Loading Time (hours)", 
                                yaxis_title="Anomaly Score")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("Severity Distribution")
                severity_df = severity_counts.reset_index()
                severity_df.columns = ['Severity', 'Count']
                fig = px.bar(severity_df, x='Severity', y='Count',
                           color='Severity',
                           labels={'Severity': 'Severity', 'Count': 'Count'},
                           color_discrete_map={
                               'Normal': '#2ca02c',
                               'Delayed': '#ff7f0e',
                               'Critical': '#d62728',
                               'Severe': '#8b0000'
                           })
                st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("Top 20 Worst Loading Time Anomalies")
            worst_anomalies = anomalies.nlargest(20, 'loading_time_hours')[[
                'VC_ITEM_DESC', 'VC_WH_CODE', 'VC_VEHICLE_NO', 'loading_time_hours',
                'anomaly_severity', 'DT_DOC_DATE', 'LOADED_DATETIME'
            ]].copy()
            worst_anomalies['loading_time_days'] = (worst_anomalies['loading_time_hours'] / 24).round(2)
            worst_anomalies['loading_time_hours'] = worst_anomalies['loading_time_hours'].round(2)
            
            st.dataframe(worst_anomalies, use_container_width=True)
            
            # Warehouse analysis
            st.subheader("Anomalies by Warehouse")
            wh_anomalies = anomalies.groupby('VC_WAREHOUSE_DESC').agg({
                'SERIAL_NO': 'count',
                'loading_time_hours': 'mean'
            }).reset_index()
            wh_anomalies.columns = ['Warehouse', 'Anomaly Count', 'Avg Loading Time (hours)']
            wh_anomalies = wh_anomalies.sort_values('Anomaly Count', ascending=False)
            
            fig = px.bar(wh_anomalies, x='Warehouse', y='Anomaly Count',
                        hover_data=['Avg Loading Time (hours)'])
            st.plotly_chart(fig, use_container_width=True)
            
            # Download
            csv = anomalies.to_csv(index=False)
            st.download_button(
                label="📥 Download All Loading Anomalies (CSV)",
                data=csv,
                file_name=f"loading_anomalies_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        else:
            st.success("✅ No significant loading time anomalies detected")
    
    # TAB 3: Shop Mismatches
    with tab3:
        st.header("🚨 Shop Mismatch Analysis")
        st.markdown("**Items ordered for Shop A but sold in Shop B**")
        
        with st.spinner("Analyzing shop mismatches..."):
            result = find_shop_mismatches(filtered_df)
        
        if result is not None and result[0] is not None:
            mismatches, mismatch_patterns = result
            
            st.error(f"⚠️ Found {len(mismatches)} shop mismatch cases")
            
            col1, col2, col3 = st.columns(3)
            
            unique_ordered_shops = mismatches['VC_SHOP_CODE'].nunique()
            unique_sold_shops = mismatches['SHOP_SOLD'].nunique()
            total_mismatch_value = mismatches['NU_SELLING_PRICE'].sum()
            
            col1.metric("Unique Ordered Shops", unique_ordered_shops)
            col2.metric("Unique Sold Shops", unique_sold_shops)
            col3.metric("Total Mismatch Value", f"₵{total_mismatch_value:,.0f}")
            
            st.markdown("---")
            
            st.subheader("Top Mismatch Patterns")
            st.markdown("**Most frequent combinations of Ordered Shop → Sold Shop**")
            
            top_patterns = mismatch_patterns.head(20).copy()
            top_patterns['Total_Value'] = top_patterns['Total_Value'].apply(lambda x: f"₵{x:,.0f}")
            st.dataframe(top_patterns, use_container_width=True)
            
            st.markdown("---")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Most Affected Ordered Shops")
                ordered_shop_counts = mismatches.groupby('VC_SHOP_CODE').size().sort_values(ascending=False).head(10)
                fig = px.bar(y=ordered_shop_counts.index, x=ordered_shop_counts.values, orientation='h',
                           labels={'x': 'Mismatch Count', 'y': 'Ordered Shop'})
                fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("Most Common Actual Sold Shops")
                sold_shop_counts = mismatches.groupby('SHOP_SOLD').size().sort_values(ascending=False).head(10)
                fig = px.bar(y=sold_shop_counts.index, x=sold_shop_counts.values, orientation='h',
                           labels={'x': 'Mismatch Count', 'y': 'Sold Shop'})
                fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            
            st.subheader("All Shop Mismatches - Detailed View")
            mismatch_display = mismatches[[
                'VC_ITEM_DESC', 'VC_SHOP_CODE', 'SHOP_SOLD', 'NU_SELLING_PRICE',
                'SERIAL_NO', 'DT_INVOICE_DATE', 'VC_WAREHOUSE_DESC'
            ]].copy()
            mismatch_display.columns = [
                'Item', 'Ordered_Shop', 'Sold_Shop', 'Price', 
                'Serial_No', 'Invoice_Date', 'Warehouse'
            ]
            mismatch_display['Price'] = mismatch_display['Price'].apply(lambda x: f"₵{x:,.2f}")
            
            st.dataframe(mismatch_display, use_container_width=True, height=400)
            
            # Sankey diagram for flow
            st.subheader("Shop Mismatch Flow Diagram")
            st.markdown("**Visual representation of items flowing from ordered shops to sold shops**")
            
            # Prepare data for Sankey
            top_20_patterns = mismatch_patterns.head(20)
            
            # Create nodes
            ordered_shops = top_20_patterns['Ordered_Shop'].unique().tolist()
            sold_shops = top_20_patterns['Sold_Shop'].unique().tolist()
            all_shops = list(set(ordered_shops + sold_shops))
            
            # Create indices
            shop_to_idx = {shop: idx for idx, shop in enumerate(all_shops)}
            
            # Prepare links
            source = [shop_to_idx[shop] for shop in top_20_patterns['Ordered_Shop']]
            target = [shop_to_idx[shop] for shop in top_20_patterns['Sold_Shop']]
            value = top_20_patterns['Count'].tolist()
            
            fig = go.Figure(data=[go.Sankey(
                node=dict(
                    pad=15,
                    thickness=20,
                    line=dict(color="black", width=0.5),
                    label=all_shops
                ),
                link=dict(
                    source=source,
                    target=target,
                    value=value
                )
            )])
            
            fig.update_layout(title_text="Top 20 Shop Mismatch Flows", font_size=10)
            st.plotly_chart(fig, use_container_width=True)
            
            # Download
            csv = mismatches.to_csv(index=False)
            st.download_button(
                label="📥 Download All Shop Mismatches (CSV)",
                data=csv,
                file_name=f"shop_mismatches_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        else:
            st.success("✅ No shop mismatches detected - all items sold in their intended shops")
    
    # TAB 4: Transit Issues
    with tab4:
        st.header("📦 Transit Time Anomalies")
        st.markdown("**Items with unusual transit times from warehouse loading to shop receipt**")
        
        with st.spinner("Detecting transit anomalies..."):
            transit_anomalies = detect_transit_anomalies(filtered_df)
        
        if transit_anomalies is not None and len(transit_anomalies) > 0:
            st.warning(f"⚠️ Found {len(transit_anomalies)} items with anomalous transit times")
            
            col1, col2, col3 = st.columns(3)
            
            avg_transit = transit_anomalies['transit_time_hours'].mean()
            max_transit = transit_anomalies['transit_time_hours'].max()
            avg_days = transit_anomalies['days_in_transit'].mean()
            
            col1.metric("Avg Transit Time", f"{avg_transit:.1f} hours")
            col2.metric("Max Transit Time", f"{max_transit:.1f} hours")
            col3.metric("Avg Transit Days", f"{avg_days:.1f} days")
            
            st.markdown("---")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Transit Time Distribution")
                fig = px.histogram(transit_anomalies, x='days_in_transit', nbins=30,
                                 labels={'days_in_transit': 'Days in Transit'})
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("Transit Time by Shop")
                shop_transit = transit_anomalies.groupby('VC_SHOP_CODE')['days_in_transit'].mean().sort_values(ascending=False).head(10)
                fig = px.bar(y=shop_transit.index, x=shop_transit.values, orientation='h',
                           labels={'x': 'Avg Days in Transit', 'y': 'Shop'})
                fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("Top 20 Longest Transit Times")
            longest_transit = transit_anomalies.nlargest(20, 'transit_time_hours')[[
                'VC_ITEM_DESC', 'VC_WH_CODE', 'VC_SHOP_CODE', 'VC_VEHICLE_NO',
                'transit_time_hours', 'days_in_transit', 'LOADED_DATETIME', 'DT_MOD_DATE'
            ]].copy()
            longest_transit['transit_time_hours'] = longest_transit['transit_time_hours'].round(2)
            longest_transit['days_in_transit'] = longest_transit['days_in_transit'].round(2)
            
            st.dataframe(longest_transit, use_container_width=True)
            
            # Download
            csv = transit_anomalies.to_csv(index=False)
            st.download_button(
                label="📥 Download All Transit Anomalies (CSV)",
                data=csv,
                file_name=f"transit_anomalies_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        else:
            st.success("✅ No significant transit time anomalies detected")

if __name__ == "__main__":
    main()
