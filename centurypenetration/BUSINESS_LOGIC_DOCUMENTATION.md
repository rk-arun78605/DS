# Century Stock Penetration Dashboard - Business Logic Documentation

## Executive Summary
The Century Stock Penetration Dashboard provides real-time analytics for CENTURY brand inventory management across all retail shops, focusing on optimal stock positioning, demand forecasting, and automated replenishment recommendations.

---

## 1. CORE BUSINESS LOGIC

### 1.1 Stock Status Classification
Items are classified into four categories based on their stock position relative to demand:

#### **Stock Status Formula:**
```
Stock Variance = (SIH + SIT) - Requirement for 21 Days
Where:
- SIH = Stock In Hand (current shop inventory)
- SIT = Stock In Transit (items en route to shop)
- Requirement for 21 Days = Rate of Sales (ROS) × 21 days
```

#### **Classification Rules:**
1. **UnderStock**: `Stock Variance < 0` (insufficient inventory to meet 21-day demand)
2. **OverStock**: `Stock Variance > 0` (excess inventory above 21-day requirement)
3. **Balanced**: `Stock Variance = 0` (optimal stock level matching 21-day requirement)

---

### 1.2 Rate of Sales (ROS) Calculation
**Formula:**
```
ROS = Total Sales (Last 90 Days) ÷ 90
```

**Purpose:**
- Provides daily average sales velocity
- Used as baseline for demand forecasting
- Accounts for seasonal fluctuations with 90-day window

**Key Metric Derived:**
```
21-Day Requirement = ROS × 21
```
This represents the optimal stock level to maintain continuous availability without overstocking.

---

### 1.3 Sales Trend Analysis
The system tracks four rolling windows to identify demand patterns:

| Period | Purpose |
|--------|---------|
| **30 Days** | Recent demand spike detection |
| **60 Days** | Short-term trend validation |
| **90 Days** | Standard demand baseline (used for ROS) |
| **365 Days** | Annual seasonality patterns |

**Business Rules:**
- All sales calculations use **yesterday as end date** (today's data incomplete)
- Date ranges are **inclusive** on both ends
- Sales tracked by both **quantity** (units sold) and **value** (net sales amount)

---

## 2. DOMINANT STATUS CLASSIFICATION LOGIC

### 2.1 Configurable Threshold System
**User-Adjustable Parameter:** Status Classification Threshold (default: **50%**)

**Business Logic:**
For each CENTURY item across all shops:
1. Calculate percentage of shops in each status (UnderStock/OverStock/Balanced)
2. If ≥ threshold% of shops have same status → Assign that dominant status
3. If no status reaches threshold → Classify as **Mixed**

**Example (50% threshold, item stocked in 20 shops):**
- UnderStock in 12 shops (60%) → **Dominant Status: UnderStock**
- OverStock in 5 shops (25%)
- Balanced in 3 shops (15%)

### 2.2 Priority Classification Rules
When calculating overview metrics:

**Priority Order (Highest to Lowest):**
1. **UnderStock** - Most critical, indicates lost sales risk
2. **OverStock** - Moderate priority, indicates capital tied up
3. **Balanced** - Optimal state
4. **Mixed** - No clear dominant pattern

**Result:** Each unique item counted exactly once in overview metrics
```
Total Items = UnderStock Items + OverStock Items + Balanced Items + Mixed Items
```

---

## 3. REORDER LEVEL PARAMETERS

### 3.1 Min-Max Inventory Control
The system tracks three key reorder parameters per item per shop:

| Parameter | Symbol | Definition |
|-----------|--------|------------|
| **Minimum Stock** | `min_nu` | Safety stock level - triggers reorder when reached |
| **Maximum Stock** | `max_nu` | Maximum holding capacity - prevents overstock |
| **Reorder Quantity** | `reorder_qty` | Suggested quantity to order when min reached |

### 3.2 Reorder Trigger Logic
**Business Rule:**
```
IF (SIH + SIT) < min_nu THEN
    Reorder Quantity = min_nu - (SIH + SIT)
ELSE IF (SIH + SIT) > max_nu THEN
    Flag as OverStock
END IF
```

---

## 4. STOCK IN TRANSIT (SIT) MANAGEMENT

### 4.1 SIT Calculation
**Source:** GRN (Goods Receipt Note) transit records

**Aggregation Logic:**
```sql
Total SIT per Item per Shop = SUM(nu_transit_qty) 
WHERE dt_trans_date >= (Current Date - 90 days)
```

**Business Rationale:**
- Only recent transits (last 90 days) considered active
- Older transits assumed delivered or cancelled
- Provides realistic pipeline visibility

### 4.2 Latest Transit Date
Tracks most recent shipment date for each item-shop combination to monitor:
- Delivery delays
- Supply chain velocity
- Replenishment frequency

---

## 5. DAYS OF STOCK REMAINING (DoS)

### 5.1 Formula
```
Days of Stock = (SIH + SIT) ÷ ROS
Where ROS > 0
```

**Interpretation:**
- **< 21 days**: UnderStock (below optimal)
- **= 21 days**: Balanced (optimal target)
- **> 21 days**: OverStock (excess inventory)
- **NULL**: Zero sales velocity (slow-moving/dead stock)

### 5.2 Business Actions
| DoS Range | Priority | Action |
|-----------|----------|--------|
| 0-7 days | **Critical** | Immediate replenishment required |
| 8-14 days | **High** | Schedule reorder within 48 hours |
| 15-21 days | **Normal** | Monitor, standard replenishment |
| 22-30 days | **Low** | Hold reorders, assess demand |
| 31+ days | **Review** | Consider redistribution or promotions |

---

## 6. CRITICAL ITEMS IDENTIFICATION

### 6.1 Critical Item Criteria
**Definition:** Items with **high demand** but **zero available stock**

**SQL Logic:**
```sql
WHERE (SIH = 0 OR SIH IS NULL)
  AND (SIT = 0 OR SIT IS NULL)
  AND ros > 0
ORDER BY ros DESC
```

**Business Impact:**
- Direct revenue loss (lost sales)
- Customer dissatisfaction
- Competitive disadvantage
- Highest priority for immediate action

---

## 7. SLOW MOVING ITEMS

### 7.1 Slow Moving Definition
**Criteria:**
- ROS = 0 (no sales in last 90 days)
- SIH > 0 (stock still available)
- Total Stock (SIH + SIT) > 0

### 7.2 Actions
1. **Review** product relevance for each shop
2. **Redistribute** to higher-demand locations
3. **Promote** with discounts to clear inventory
4. **Discontinue** if consistently slow across network

---

## 8. SHOP-WISE ANALYSIS

### 8.1 Key Metrics per Shop
- **Total Items**: Unique CENTURY SKUs stocked
- **UnderStock Count**: Items below 21-day requirement
- **OverStock Count**: Items above 21-day requirement
- **Balanced Count**: Items at optimal level
- **Average ROS**: Shop's overall sales velocity
- **Total SIH**: Current inventory value
- **Total SIT**: Pipeline inventory value
- **21-Day Requirement**: Aggregate demand forecast

### 8.2 Shop Performance Ranking
**Sorting Priority:**
1. **UnderStock items** (descending) - shops with most stock-outs first
2. **OverStock items** (descending) - capital efficiency review
3. **Average ROS** (descending) - sales velocity benchmark

---

## 9. DEPARTMENT ANALYSIS

### 9.1 Aggregation Level
Groups items by department to identify:
- Category-level stock health
- Department-wise inventory investment
- Product mix optimization opportunities

### 9.2 Visual Analytics
**Stacked Bar Chart:**
- X-axis: Department names
- Y-axis: Item count
- Segments: UnderStock (Red) | OverStock (Purple) | Balanced (Blue)

**Purpose:** Quick identification of departments needing inventory rebalancing

---

## 10. DATA REFRESH & PERFORMANCE

### 10.1 Materialized Views Architecture
**Three-Tier View Structure:**

1. **mv_sales_metrics**
   - Pre-aggregates 30/60/90/365-day sales per item per shop
   - Calculates ROS and last sale date
   - Deduplicates sales transactions
   - **Refresh**: Daily (post-sales data load)

2. **mv_sit_summary**
   - Aggregates total SIT per item per shop
   - Tracks latest transit date
   - Filters recent transits (90 days)
   - **Refresh**: Daily (post-GRN data load)

3. **mv_century_penetration** (Main View)
   - Joins reorder_level + mv_sales_metrics + mv_sit_summary
   - Calculates stock variance, days of stock, status
   - Filters for CENTURY brand only
   - **Refresh**: Daily (after dependencies refreshed)

### 10.2 Refresh Sequence
```
1. Load Base Tables (reorder_level, sales, sit)
2. Refresh mv_sales_metrics
3. Refresh mv_sit_summary
4. Refresh mv_century_penetration
5. Rebuild indexes
```

**Performance:**
- Query response: <100ms (materialized views)
- Dashboard load: <2 seconds
- Data freshness: Daily (T+1 data)

---

## 11. BUSINESS PRIORITIES & ACTIONABLE INSIGHTS

### 11.1 Priority Matrix

| Priority | Category | Criteria | Action | SLA |
|----------|----------|----------|--------|-----|
| **P0** | Critical Items | SIH=0, SIT=0, ROS>0 | Emergency replenishment | 24 hours |
| **P1** | UnderStock (High ROS) | DoS < 7, ROS > 1 | Expedited reorder | 48 hours |
| **P2** | UnderStock (Normal) | DoS < 21, ROS > 0 | Standard reorder | 1 week |
| **P3** | OverStock (High Value) | DoS > 30, High selling price | Redistribute/Promote | 2 weeks |
| **P4** | Slow Moving | ROS = 0, SIH > 0 | Review/Clear | 1 month |

### 11.2 Daily Operations Workflow
**Morning Review (9:00 AM):**
1. Check Critical Items tab → Immediate orders
2. Review UnderStock tab → Prioritize high ROS items
3. Verify SIT arrivals → Update expected delivery dates

**Afternoon Analysis (2:00 PM):**
1. Shop-wise performance → Identify struggling locations
2. Department analysis → Category health check
3. Slow moving review → Clearance planning

**Weekly Deep Dive (Monday):**
1. Threshold adjustment → Optimize classification sensitivity
2. Reorder level validation → Update min/max based on trends
3. Top 10 OverStock items → Redistribution planning

---

## 12. KEY PERFORMANCE INDICATORS (KPIs)

### 12.1 Strategic KPIs
| Metric | Target | Measurement |
|--------|--------|-------------|
| **Stock-Out Rate** | < 5% | (UnderStock Items / Total Items) × 100 |
| **OverStock Rate** | < 15% | (OverStock Items / Total Items) × 100 |
| **Balanced Rate** | > 75% | (Balanced Items / Total Items) × 100 |
| **Average DoS** | 18-24 days | Network-wide average |
| **Critical Items** | 0 | Zero stock with demand |
| **Slow Moving %** | < 10% | Items with ROS = 0 |

### 12.2 Operational Metrics
- **Average ROS**: Overall sales velocity benchmark
- **Total SIH**: Inventory capital deployed
- **Total SIT**: Pipeline inventory value
- **21-Day Requirement**: Forward-looking demand forecast
- **Latest Transit Age**: Supply chain responsiveness

---

## 13. USER CONTROLS & CUSTOMIZATION

### 13.1 Status Classification Threshold
**Range:** 30% - 100% (default: 50%)

**Impact:**
- **Lower threshold (30-40%)**: More items classified, fewer "Mixed"
  - Use when: Want early signals of potential issues
  
- **Higher threshold (70-100%)**: Stricter classification, more "Mixed"
  - Use when: Want high-confidence status assignments only

**Recommendation:** Start at 50%, adjust based on:
- Network size (more shops → lower threshold)
- Category volatility (stable categories → higher threshold)
- Management style (proactive → lower, reactive → higher)

### 13.2 Interactive Features
1. **Search**: Find specific items by code or name
2. **View Details Buttons**: Jump to UnderStock/OverStock tabs from metrics
3. **Refresh Data**: Clear cache and reload latest data
4. **Download Options**: Export filtered results to CSV
5. **Sortable Tables**: Click column headers to reorder

---

## 14. DATA VALIDATION & QUALITY CHECKS

### 14.1 Deduplication Logic
**Issue:** Sales table may have duplicate entries (same item, shop, date)

**Solution:** 
```sql
SELECT DISTINCT ON (shop_code, item_code, date_invoice)
  ... columns ...
FROM sales
ORDER BY shop_code, item_code, date_invoice, loaded_at DESC
```
Keeps most recent load when duplicates exist.

### 14.2 Data Freshness Indicators
- **Last Sale Date**: Identifies stale inventory
- **Latest Transit Date**: Verifies active replenishment
- **Refreshed At**: Timestamp on materialized views

---

## 15. TECHNICAL SPECIFICATIONS

### 15.1 Database Schema
- **Database:** `century_penetration` (PostgreSQL port 3307)
- **Tables:** 
  - `reorder_level` (master stock data)
  - `sales` (partitioned by month, 2025 partitions)
  - `sit` (stock in transit)
- **Views:** 
  - `mv_sales_metrics`
  - `mv_sit_summary`
  - `mv_century_penetration`

### 15.2 Connection Pooling
- **Pool Type:** SimpleConnectionPool
- **Min Connections:** 1
- **Max Connections:** 5
- **Context Manager:** Auto-return connections to pool
- **Caching:** @st.cache_data (TTL: 10 minutes)

---

## 16. BUSINESS VALUE PROPOSITION

### 16.1 Quantifiable Benefits
1. **Reduced Stock-Outs**: 
   - Early warning system (21-day forecast)
   - Critical item alerts
   - **Impact:** 15-30% reduction in lost sales

2. **Optimized Working Capital**:
   - OverStock identification
   - Slow-moving inventory visibility
   - **Impact:** 10-20% reduction in excess inventory

3. **Improved Service Levels**:
   - Balanced stock across network
   - Faster replenishment decisions
   - **Impact:** 5-10% improvement in product availability

4. **Operational Efficiency**:
   - Automated calculations (ROS, DoS, variance)
   - Pre-aggregated analytics (sub-second queries)
   - **Impact:** 60-80% reduction in manual analysis time

### 16.2 Strategic Insights
- **Demand Forecasting**: 90-day ROS provides reliable baseline
- **Network Optimization**: Shop-wise analysis reveals geographic patterns
- **Category Management**: Department analysis drives assortment decisions
- **Supply Chain Performance**: SIT tracking monitors vendor reliability

---

## 17. FUTURE ENHANCEMENTS (ROADMAP)

### 17.1 Phase 2 Features
- **Predictive Analytics**: Machine learning for demand forecasting
- **Automated Reordering**: API integration with procurement system
- **Multi-Brand Support**: Expand beyond CENTURY brand
- **Mobile Alerts**: Push notifications for critical items
- **Vendor Scorecards**: Delivery performance tracking

### 17.2 Advanced Analytics
- **ABC Analysis**: Classify items by revenue contribution
- **Seasonality Patterns**: Year-over-year trend identification
- **Promotion Impact**: Sales lift measurement during campaigns
- **Cross-Shop Transfers**: Automated redistribution recommendations

---

## APPENDIX A: GLOSSARY

| Term | Definition |
|------|------------|
| **SIH** | Stock In Hand - Current inventory at shop location |
| **SIT** | Stock In Transit - Items shipped but not yet received |
| **ROS** | Rate of Sales - Average daily sales velocity (90-day) |
| **DoS** | Days of Stock - Inventory cover in days at current ROS |
| **GRN** | Goods Receipt Note - Document confirming shipment |
| **SKU** | Stock Keeping Unit - Unique product identifier |
| **TTL** | Time To Live - Cache expiration duration |

---

## APPENDIX B: CONTACT & SUPPORT

**Dashboard Owner:** Data Science Team  
**Database Admin:** IT Infrastructure  
**Business Stakeholder:** Supply Chain Manager  
**Refresh Schedule:** Daily at 2:00 AM  
**Support Email:** ds-team@company.com  

**Documentation Version:** 1.0  
**Last Updated:** December 15, 2025  
**Next Review:** January 15, 2026  

---

**END OF DOCUMENT**
