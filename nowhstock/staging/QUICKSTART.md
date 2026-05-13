# 🚀 QUICK START - Staging Environment

## ⚡ 3-Minute Setup

### 1. Create Staging Views (5-10 min)
```batch
cd "D:\Dashboard Code\NO_WH\DS\nowhstock\staging\batch"
1_setup_staging.bat
```
Wait for completion... ☕

### 2. Launch Staging Dashboard
```batch
2_run_staging_dashboard.bat
```
Opens at: http://localhost:8502

### 3. Compare with Production
Open both dashboards side-by-side:
- **Production:** http://localhost:8501
- **Staging:** http://localhost:8502

---

## 🔍 Quick Test

### Filter by Priority Shop

**In Production (8501):**
1. Filter → Shop → Select "SPN"
2. Click "Get Recommendations"
3. Result: `0 recommendations` (SPN cannot be source)

**In Staging (8502):**
1. Filter → Shop → Select "SPN"
2. Click "Get Recommendations"
3. Result: `~15,000+ recommendations` (SPN as source! 🆕)

---

## ✅ Quick Validation

Run comparison script:
```batch
4_compare_staging_vs_production.bat
```

**Expected Output:**
```
Total Recommendations:
  Production: 500,993
  Staging:    789,663  ✅ Higher

Priority Shop Sources:
  Production: 0
  Staging:    11  ✅ Priority shops as sources

Same-Shop Transfers:
  Production: 0
  Staging:    0   ✅ Correct (no same-shop)
```

---

## 📊 Quick Stats

| Metric | Production | Staging | Status |
|--------|-----------|---------|---------|
| Total Recs | 500K | 790K | ✅ +58% |
| Priority Sources | 0 | 11 | ✅ NEW |
| Same-Shop | 0 | 0 | ✅ Safe |

---

## 🎯 Next Steps

1. **Test for 1 week** - Use staging dashboard daily
2. **Validate transfers** - Review priority-to-priority recommendations
3. **Get approval** - Business team sign-off
4. **Promote to production** - Run `5_promote_to_production.bat`

---

## 📞 Need Help?

**Full Documentation:**
- `README.md` - Complete guide
- `VISUAL_COMPARISON.md` - Side-by-side examples

**Quick Commands:**
```batch
# Refresh data
3_refresh_staging_views.bat

# Compare environments
4_compare_staging_vs_production.bat

# Promote (after approval)
5_promote_to_production.bat
```

**Database Access:**
```batch
psql -U postgres -d salesdata -p 3307
\d mv_recommendations_complete_staging
```

---

## ⚠️ Important Notes

- ✅ **Production is SAFE** - Completely isolated
- ✅ **Different ports** - 8501 (prod) vs 8502 (staging)
- ✅ **Separate views** - `_staging` suffix on all objects
- ❌ **Don't promote without testing** - Use for minimum 1 week

---

**Setup Time:** ~10 minutes  
**Status:** Ready to test  
**Updated:** December 12, 2025
