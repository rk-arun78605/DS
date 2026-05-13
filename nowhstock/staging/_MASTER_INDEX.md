# 🧪 STAGING ENVIRONMENT - MASTER INDEX

## 📍 Location
```
D:\Dashboard Code\NO_WH\DS\nowhstock\staging\
```

## 📋 Complete File Inventory

### 🎯 Core Application
| File | Size | Purpose |
|------|------|---------|
| `nowhstock_ds_STAGING.py` | 212 KB | Streamlit dashboard (port 8502) |

### 💾 Database Setup
| File | Size | Purpose |
|------|------|---------|
| `sql/create_staging_views.sql` | 15 KB | Creates all staging views & indexes |

### 🔧 Automation Scripts
| File | Size | Purpose |
|------|------|---------|
| `batch/1_setup_staging.bat` | 2.6 KB | Initial setup (run FIRST) |
| `batch/2_run_staging_dashboard.bat` | 1.1 KB | Launch dashboard |
| `batch/3_refresh_staging_views.bat` | 1.8 KB | Refresh data |
| `batch/4_compare_staging_vs_production.bat` | 3.5 KB | Compare environments |
| `batch/5_promote_to_production.bat` | 4.9 KB | Move to production |

### 📚 Documentation
| File | Size | Purpose |
|------|------|---------|
| `QUICKSTART.md` | 2.6 KB | 3-minute setup guide |
| `README.md` | 10.7 KB | Complete detailed guide |
| `VISUAL_COMPARISON.md` | 13 KB | Side-by-side examples |
| `_STAGING_SUMMARY.md` | 11 KB | Quick reference summary |
| `_MASTER_INDEX.md` | This file | File inventory |

---

## 🚀 EXECUTION ORDER

### First Time Setup
```
1. Read Documentation
   └─ Start with: QUICKSTART.md (3 minutes)
   └─ Full details: README.md (15 minutes)
   └─ Visual examples: VISUAL_COMPARISON.md

2. Create Staging Environment
   └─ Run: batch/1_setup_staging.bat (10 minutes)
   └─ Creates all database views and indexes

3. Launch Dashboard
   └─ Run: batch/2_run_staging_dashboard.bat
   └─ Opens at: http://localhost:8502

4. Test Thoroughly
   └─ Compare with production (http://localhost:8501)
   └─ Validate priority-to-priority transfers
   └─ Run: batch/4_compare_staging_vs_production.bat
```

### Daily Testing
```
1. Refresh Data (if needed)
   └─ Run: batch/3_refresh_staging_views.bat

2. Test Dashboard
   └─ Run: batch/2_run_staging_dashboard.bat

3. Validate Results
   └─ Run: batch/4_compare_staging_vs_production.bat
```

### After Approval
```
1. Final Validation
   └─ Run full checklist from README.md

2. Promote to Production
   └─ Run: batch/5_promote_to_production.bat
   └─ Requires typing "YES" to confirm
```

---

## 📊 What Gets Created

### Database Objects

#### Materialized Views
```sql
mv_slow_fast_moving_summary_staging       -- Movement categorization
mv_recommendations_complete_staging        -- Main recommendations (with priority-to-priority)
```

#### Indexes (7 total)
```sql
idx_mv_slow_fast_staging_item_shop        -- Slow/fast lookup
idx_mv_slow_fast_staging_category         -- Category filter
idx_mv_recs_staging_item_dest             -- Main lookup
idx_mv_recs_staging_source                -- Source filter
idx_mv_recs_staging_dest                  -- Destination filter
idx_mv_recs_staging_item                  -- Item filter
idx_mv_recs_staging_qty                   -- Quantity sorting
idx_mv_recs_staging_groups                -- Group filter
idx_mv_recs_staging_subgroup              -- Subgroup filter
```

---

## 🔐 Safety Checklist

### Before Running Anything
- [x] Production code located at: `D:\Dashboard Code\NO_WH\DS\nowhstock_ds.py`
- [x] Production uses: `mv_recommendations_complete` (NO _staging suffix)
- [x] Staging completely isolated in: `staging/` subfolder
- [x] Staging uses different port: 8502 (production uses 8501)
- [x] All staging objects have `_staging` suffix

### Isolation Verification
```sql
-- Check both views exist independently
SELECT COUNT(*) FROM mv_recommendations_complete;           -- Production
SELECT COUNT(*) FROM mv_recommendations_complete_staging;   -- Staging

-- Verify different counts
-- Production: ~500K, Staging: ~790K
```

---

## 📱 Quick Access Commands

### PowerShell (from any location)
```powershell
# Setup
cd "D:\Dashboard Code\NO_WH\DS\nowhstock\staging\batch"
.\1_setup_staging.bat

# Run dashboard
.\2_run_staging_dashboard.bat

# Compare
.\4_compare_staging_vs_production.bat

# Promote (AFTER APPROVAL)
.\5_promote_to_production.bat
```

### PostgreSQL
```sql
-- Access database
psql -U postgres -d salesdata -p 3307

-- Check staging views
\d mv_recommendations_complete_staging
\d mv_slow_fast_moving_summary_staging

-- Compare counts
SELECT 'Production' as env, COUNT(*) FROM mv_recommendations_complete
UNION ALL
SELECT 'Staging' as env, COUNT(*) FROM mv_recommendations_complete_staging;
```

---

## 🎯 Key Differences - Quick Reference

| Aspect | Production | Staging |
|--------|-----------|---------|
| **View Name** | `mv_recommendations_complete` | `mv_recommendations_complete_staging` |
| **Priority as Sources** | ❌ NO | ✅ YES |
| **Total Recs** | ~500K | ~790K |
| **Port** | 8501 | 8502 |
| **File** | `nowhstock_ds.py` | `staging/nowhstock_ds_STAGING.py` |
| **SQL** | `NowhStock_mv_recommendations_complete.sql` | `staging/sql/create_staging_views.sql` |

---

## 📖 Documentation Guide

### For Quick Setup (3-5 minutes)
→ Read: **QUICKSTART.md**
- Minimal commands
- Essential steps only
- Quick validation

### For Complete Understanding (15-20 minutes)
→ Read: **README.md**
- Full architecture
- All features explained
- Complete validation checklist
- Troubleshooting guide
- Promotion process

### For Visual Examples (10 minutes)
→ Read: **VISUAL_COMPARISON.md**
- Side-by-side comparisons
- Diagram examples
- Dashboard screenshots (text)
- Example scenarios

### For Quick Reference (5 minutes)
→ Read: **_STAGING_SUMMARY.md**
- Key changes summary
- Safety features
- Quick commands
- Validation checklist

### For File Navigation (2 minutes)
→ Read: **_MASTER_INDEX.md** (this file)
- Complete file list
- Execution order
- Quick access commands

---

## 🆘 Emergency Contacts

### Database Issues
```sql
-- Access database
psql -U postgres -d salesdata -p 3307

-- Drop staging views (if needed to recreate)
DROP MATERIALIZED VIEW IF EXISTS mv_recommendations_complete_staging CASCADE;
DROP MATERIALIZED VIEW IF EXISTS mv_slow_fast_moving_summary_staging CASCADE;

-- Then re-run: 1_setup_staging.bat
```

### Dashboard Issues
```powershell
# Check if port 8502 is in use
Get-NetTCPConnection -LocalPort 8502

# Kill Streamlit process
Get-Process streamlit | Stop-Process -Force

# Restart dashboard
.\2_run_staging_dashboard.bat
```

### Code Issues
```powershell
# Restore original staging code (if modified)
Copy-Item "D:\Dashboard Code\NO_WH\DS\nowhstock_ds.py" `
          "D:\Dashboard Code\NO_WH\DS\nowhstock\staging\nowhstock_ds_STAGING.py" -Force

# Then manually update view names again
```

---

## ✅ Pre-Flight Checklist

### Before First Run
- [ ] Read QUICKSTART.md
- [ ] Understand what will be created
- [ ] Confirm production is NOT affected
- [ ] Have database credentials ready
- [ ] Have 10 minutes for setup

### After Setup
- [ ] Staging dashboard opens successfully
- [ ] Can login with credentials
- [ ] Filters populate correctly
- [ ] Can see priority shops in source dropdown
- [ ] Compare script shows expected differences

### Before Promotion
- [ ] Tested for minimum 1 week
- [ ] Business team approval obtained
- [ ] All validation checks pass
- [ ] Users notified of change
- [ ] Backup strategy confirmed

---

## 📞 Support Resources

### Documentation Files (in order of importance)
1. **QUICKSTART.md** - Start here
2. **README.md** - Full guide
3. **VISUAL_COMPARISON.md** - Examples
4. **_STAGING_SUMMARY.md** - Quick reference
5. **_MASTER_INDEX.md** - This file

### Batch Scripts (in execution order)
1. **1_setup_staging.bat** - Initial setup
2. **2_run_staging_dashboard.bat** - Launch app
3. **4_compare_staging_vs_production.bat** - Validate
4. **3_refresh_staging_views.bat** - Refresh data
5. **5_promote_to_production.bat** - Promote (LAST)

### SQL Files
1. **create_staging_views.sql** - Complete database setup

---

## 🎉 Success Indicators

### ✅ Setup Successful When:
- Staging views created (~10 min)
- Dashboard opens on port 8502
- Can filter by priority shop as source
- Compare script shows staging > production
- No error messages

### ✅ Testing Successful When:
- Used for 1+ weeks without issues
- Business team approves transfers
- All validation checks pass
- Performance acceptable (<100ms)
- Users comfortable with changes

### ✅ Production Ready When:
- All checklists completed
- Formal approval obtained
- Backup confirmed
- Users notified
- Rollback plan ready

---

**Created:** December 12, 2025  
**Total Files:** 13 (1 app + 1 SQL + 5 batch + 5 docs + this index)  
**Total Size:** ~276 KB  
**Status:** ✅ Complete & Ready for Testing  
**Production Impact:** ⚠️ ZERO (Completely Isolated)
