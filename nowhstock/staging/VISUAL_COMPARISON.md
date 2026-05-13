# 📊 STAGING vs PRODUCTION - Visual Comparison Guide

## 🎯 Quick Overview

| Aspect | Production (Current) | Staging (Testing) |
|--------|---------------------|-------------------|
| **View Name** | `mv_recommendations_complete` | `mv_recommendations_complete_staging` |
| **Priority Shops as Sources** | ❌ NO | ✅ YES |
| **Total Recommendations** | ~500,000 | ~790,000 (+58%) |
| **Dashboard Port** | 8501 | 8502 |
| **Production Impact** | N/A | ⚠️ ZERO (isolated) |

---

## 🔄 Key Business Logic Change

### Production (OLD)
```
┌─────────────────────────────────────────┐
│  NON-PRIORITY SHOPS (Sources)           │
│  ACC, ACH, ADB, AFL, etc. (68 shops)    │
│          ↓ Transfers                    │
│  PRIORITY SHOPS (Destinations ONLY)     │
│  SPN, MSS, LFS, M03, KAS, etc. (11)    │
└─────────────────────────────────────────┘
```
**Rule:** Priority shops CANNOT be sources

### Staging (NEW)
```
┌─────────────────────────────────────────┐
│  ALL SHOPS (Sources)                    │
│  Priority (11) + Non-Priority (68)      │
│          ↓ Transfers                    │
│  PRIORITY SHOPS (Destinations)          │
│  SPN, MSS, LFS, M03, KAS, etc. (11)    │
│                                         │
│  🆕 Priority → Priority Allowed         │
│  (But NOT same shop, e.g., SPN → SPN)  │
└─────────────────────────────────────────┘
```
**New Rule:** Priority shops CAN transfer to OTHER priority shops

---

## 📈 Example Scenarios

### Scenario 1: Item ABC123 at SPN (Priority Shop)

#### Production
```
SPN (Source): Stock = 500, Sales 30d = 150
Result: ❌ BLOCKED - Priority shops cannot be sources
         Item sits at SPN even if other priority shops need it
```

#### Staging
```
SPN (Source): Stock = 500, Sales 30d = 150
Available to transfer: 350 units (500 - 150)

Destinations:
  MSS needs 100 → Transfer 100 ✅
  LFS needs 50  → Transfer 50  ✅
  M03 needs 200 → Transfer 200 ✅

Result: ✅ ALLOWED - SPN can transfer to other priority shops
        Better inventory distribution across priority network
```

---

### Scenario 2: Priority-to-Priority Transfer Matrix

#### Example: Item XYZ789

**Staging enables these transfers:**

| Source → Dest | MSS | LFS | M03 | KAS | MM1 |
|--------------|-----|-----|-----|-----|-----|
| **SPN** | ✅ | ✅ | ✅ | ✅ | ✅ |
| **MSS** | ❌ | ✅ | ✅ | ✅ | ✅ |
| **LFS** | ✅ | ❌ | ✅ | ✅ | ✅ |
| **M03** | ✅ | ✅ | ❌ | ✅ | ✅ |

✅ = Transfer allowed  
❌ = Blocked (same shop)

**Production:** All cells would be ❌ (priority shops excluded from sources)

---

## 🔍 Dashboard Comparison

### Login Screen
```
Production:                    Staging:
┌─────────────────────┐       ┌─────────────────────┐
│ Inventory Pulse     │       │ 🧪 STAGING          │
│ NO_WH               │       │ Inventory Pulse     │
│                     │       │ NO_WH               │
│ [Login Form]        │       │ [Login Form]        │
└─────────────────────┘       └─────────────────────┘
```

### Filter Panel
```
Production:                    Staging:
┌─────────────────────┐       ┌─────────────────────┐
│ Filters             │       │ Filters             │
│ ├─ Group: All       │       │ ├─ Group: All       │
│ ├─ SubGroup: All    │       │ ├─ SubGroup: All    │
│ ├─ Item: All        │       │ ├─ Item: All        │
│ └─ Shop: All        │       │ └─ Shop: All        │
│                     │       │                     │
│ Source Shops: 68    │       │ Source Shops: 79    │
│ (No priority shops) │       │ (Includes priority) │
└─────────────────────┘       └─────────────────────┘
```

### Results Table
```
Production:                           Staging:
┌─────────────────────────────────┐  ┌─────────────────────────────────┐
│ Source  Dest   Item    Qty      │  │ Source  Dest   Item    Qty      │
│ ACC  → SPN  ITEM001   50        │  │ ACC  → SPN  ITEM001   50        │
│ ACH  → MSS  ITEM002   30        │  │ ACH  → MSS  ITEM002   30        │
│ ADB  → LFS  ITEM003   20        │  │ ADB  → LFS  ITEM003   20        │
│ ...                              │  │ SPN  → MSS  ITEM004   100  🆕   │
│                                  │  │ MSS  → LFS  ITEM005   75   🆕   │
│                                  │  │ LFS  → M03  ITEM006   60   🆕   │
│ Total: 500,993                   │  │ Total: 789,663                  │
└─────────────────────────────────┘  └─────────────────────────────────┘
                                     
                                     🆕 = Priority-to-Priority
```

---

## 📊 Metrics Dashboard

### Summary Stats

#### Production
```
┌──────────────────────────────────┐
│  Total Recommendations: 500,993  │
│  Unique Items: 45,234            │
│  Unique Sources: 68              │
│  Priority Sources: 0             │
│  Destination Shops: 11           │
└──────────────────────────────────┘
```

#### Staging
```
┌──────────────────────────────────┐
│  Total Recommendations: 789,663  │
│  Unique Items: 52,108            │
│  Unique Sources: 79 (+11)        │
│  Priority Sources: 11 🆕         │
│  Destination Shops: 11           │
│                                  │
│  Priority-to-Priority: 288,670   │
│  (36.5% of total)                │
└──────────────────────────────────┘
```

---

## 🎨 Color-Coded Priority Shops

### In Staging Dashboard

When viewing results, priority-to-priority transfers can be highlighted:

```
Source Shop        Type                  Indicator
─────────────────  ──────────────────── ─────────
ACC                Non-Priority          (normal)
SPN                Priority → Priority   🔵 BLUE
MSS                Priority → Priority   🔵 BLUE
LFS                Priority → Priority   🔵 BLUE
```

**Suggestion:** Add visual indicators in dashboard to quickly identify new transfer types

---

## 🧪 Testing Workflow

### Step-by-Step Visual Guide

```
1. Setup Staging
   └─ Run: 1_setup_staging.bat
      ├─ Creates staging views
      ├─ Creates indexes
      └─ ✅ Verified

2. Launch Both Dashboards
   ├─ Production: localhost:8501
   └─ Staging:    localhost:8502

3. Side-by-Side Testing
   ┌─────────────────┬─────────────────┐
   │ Production 8501 │ Staging 8502    │
   ├─────────────────┼─────────────────┤
   │ Filter: SPN     │ Filter: SPN     │
   │ Results: 0      │ Results: 1,234  │
   │ (No sources)    │ (As source! 🆕) │
   └─────────────────┴─────────────────┘

4. Validate Results
   └─ Run: 4_compare_staging_vs_production.bat
      ├─ Total counts
      ├─ Priority sources
      ├─ Same-shop check (should be 0)
      └─ Sample transfers

5. Business Approval
   ├─ Review sample transfers
   ├─ Validate business logic
   └─ Sign off on changes

6. Promote to Production
   └─ Run: 5_promote_to_production.bat
      ├─ Backup production
      ├─ Replace with staging
      ├─ Rename indexes
      └─ ✅ Complete
```

---

## ⚠️ What Could Go Wrong?

### Scenario: Same-Shop Transfers

**Wrong Implementation:**
```sql
-- Missing WHERE clause
FROM sources src
INNER JOIN destinations dst 
  ON TRIM(UPPER(src.item_code)) = TRIM(UPPER(dst.item_code))
-- ❌ Allows SPN → SPN
```

**Correct Implementation:**
```sql
-- With WHERE clause
FROM sources src
INNER JOIN destinations dst 
  ON TRIM(UPPER(src.item_code)) = TRIM(UPPER(dst.item_code))
WHERE TRIM(UPPER(src.shop_code)) != TRIM(UPPER(dst.shop_code))
-- ✅ Blocks SPN → SPN
```

**Detection:**
```batch
4_compare_staging_vs_production.bat
# Shows: Same-Shop Transfers (should be 0)
```

---

### Scenario: Capacity Violations

**Risk:**
Source shop transfers MORE than available stock

**Protection:**
```sql
-- Cumulative allocation tracking
SUM(allocated_qty) OVER (
    PARTITION BY item_code, dest_shop
    ORDER BY priority_rank, source_grn_age DESC
) <= dest_capacity
```

**Verification:**
```sql
SELECT 
    item_code, 
    dest_shop,
    SUM(recommended_qty) as total,
    MAX(dest_sales_used) as cap
FROM mv_recommendations_complete_staging
GROUP BY item_code, dest_shop
HAVING SUM(recommended_qty) > MAX(dest_sales_used);
-- Should return 0 rows
```

---

## 📋 Checklist - Before Promotion

### Visual Checks

- [ ] **Dashboard Title:** Shows "🧪 STAGING" badge
- [ ] **Port Number:** Staging runs on 8502, production on 8501
- [ ] **Filter Results:** Staging shows more results with same filters
- [ ] **Source Shops:** Priority shops appear in staging source dropdown
- [ ] **Results Table:** Priority-to-priority transfers visible
- [ ] **Metrics:** Total recommendations higher in staging
- [ ] **Excel Export:** Works without errors (with limit)

### Database Checks

- [ ] **View Exists:** `mv_recommendations_complete_staging` found
- [ ] **Row Count:** >500,000 recommendations
- [ ] **Priority Sources:** 1-11 priority shops as sources
- [ ] **Same-Shop:** 0 same-shop transfers
- [ ] **Indexes:** All 7 staging indexes created
- [ ] **Performance:** Queries <100ms with indexes
- [ ] **Capacity:** No violations found

### Business Checks

- [ ] **Sample Transfers:** Reviewed by business team
- [ ] **Priority Logic:** Makes sense operationally
- [ ] **Stock Levels:** Realistic and achievable
- [ ] **Expiry Rules:** Still blocking <30 days
- [ ] **FEFO Order:** Oldest stock prioritized
- [ ] **Edge Cases:** Tested (no sales, no GRN, expired)

---

## 🎉 Success Criteria

### Staging is ready for production when:

✅ All validation checks pass  
✅ Business team approves transfers  
✅ Testing period complete (1 week minimum)  
✅ No same-shop transfers found  
✅ Performance acceptable (<100ms)  
✅ Users trained on new logic  
✅ Backup strategy confirmed  
✅ Rollback plan documented  

---

## 📞 Quick Help

### "How do I see priority-to-priority transfers?"
```
1. Open staging dashboard (8502)
2. Filter by Shop: Select priority shop (e.g., SPN)
3. Results will show SPN as SOURCE
4. Compare with production (8501) - should show 0 results
```

### "How do I verify same-shop blocking?"
```batch
cd D:\Dashboard Code\NO_WH\DS\nowhstock\staging\batch
4_compare_staging_vs_production.bat
# Look for: Same-Shop Transfers (should be 0)
```

### "How do I refresh staging data?"
```batch
cd D:\Dashboard Code\NO_WH\DS\nowhstock\staging\batch
3_refresh_staging_views.bat
```

### "How do I compare total counts?"
```sql
psql -U postgres -d salesdata -p 3307
SELECT 
  'Production' as env, COUNT(*) 
FROM mv_recommendations_complete
UNION ALL
SELECT 
  'Staging' as env, COUNT(*) 
FROM mv_recommendations_complete_staging;
```

---

**Last Updated:** December 12, 2025  
**Status:** Ready for Testing  
**Next Review:** After 1 week of testing
