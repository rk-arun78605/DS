# 🎯 BARCODE MATCHER - QUICK START GUIDE

## ✅ What You Have Now

Complete barcode matching system with **2 files and 3 operations**:

### 📁 File 1: Master Data (Daily Updated)
**Location:** `D:\Dashboard Code\NO_WH\DS\barcode_matcher\sample_data\barcodedata.csv`

**Format:**
```csv
VC_ITEM_BARCODE,VC_ITEM_CODE
038841004018,D448
607481821827,1683
```

**Current Status:** ✅ **229,017 barcodes loaded** in database

---

### 📁 File 2: User Upload (Anytime)
**Single column only:**
```csv
barcode
038841004018
INVALID123
607481821827
```

**Upload via:** http://localhost:8503

---

## 🚀 Three Simple Operations

### 1️⃣ Daily Master Update (Morning - Automated)
```batch
cd barcode_matcher\batch
load_barcode_master.bat
```

**What it does:**
- Reads `barcodedata.csv` (or `.xlsx`)
- Compares with database
- Adds **ONLY NEW** barcodes
- Logs everything

**Time:** ~30 seconds for 1000 new barcodes

---

### 2️⃣ Run Dashboard (Once per day)
```batch
cd barcode_matcher\batch
run_barcode_matcher.bat
```

**Opens:** http://localhost:8503

---

### 3️⃣ User Workflow (All Day)
1. User uploads CSV (single `barcode` column)
2. System matches instantly
3. User downloads results:
   ```csv
   barcode,item_code,match_status
   038841004018,D448,Matched
   INVALID123,,Not Found
   ```

---

## 📊 Current Database

```
Total Barcodes: 229,017
Database: salesdata (port 3307)
Table: barcode_item_master
```

**Check status:**
```batch
cd barcode_matcher\batch
check_status.bat
```

---

## 🎯 Key Features

✅ **Single column upload** - Users only need barcode  
✅ **All rows preserved** - Matched + unmatched  
✅ **Blank for no-match** - Empty item_code if not found  
✅ **Daily auto-update** - Only new entries loaded  
✅ **Fast matching** - <1 second for 1000 barcodes  
✅ **Audit trail** - All uploads logged  

---

## 📝 Daily Workflow Example

**Morning (6:00 AM):**
```
Supplier updates: barcodedata.csv (100 new barcodes)
Run: load_barcode_master.bat
Result: 100 NEW barcodes added to database
```

**During Day:**
```
User 1 uploads: 500 barcodes → 485 matched, 15 not found
User 2 uploads: 200 barcodes → 200 matched
User 3 uploads: 1000 barcodes → 950 matched, 50 not found
```

**All happens via web dashboard - No manual intervention needed!**

---

## 🔧 Files Created

```
barcode_matcher/
├── barcode_matcher_app.py               ← Streamlit app
├── sql/create_barcode_tables_clean.sql  ← Database schema
├── batch/
│   ├── load_barcode_master.py          ← Daily loader
│   ├── load_barcode_master.bat         ← Run daily
│   ├── run_barcode_matcher.bat         ← Run dashboard
│   ├── check_status.bat                ← Check database
│   └── setup_barcode_system_updated.bat ← One-time setup (DONE)
├── sample_data/
│   ├── barcodedata.csv                 ← Master data (229K barcodes)
│   └── user_upload_sample.csv          ← Example user file
└── README.md                            ← Full documentation
```

---

## ⚡ Quick Commands

**Check database:**
```batch
cd barcode_matcher\batch
check_status.bat
```

**Run daily load:**
```batch
load_barcode_master.bat
```

**Start dashboard:**
```batch
run_barcode_matcher.bat
```

**View log:**
```batch
type ..\sample_data\barcode_load_log.txt
```

---

## 🎉 You're Ready!

1. ✅ Database created with 229,017 barcodes
2. ✅ Daily loader ready (`load_barcode_master.bat`)
3. ✅ Dashboard ready (`run_barcode_matcher.bat`)
4. ✅ User upload format defined (single `barcode` column)

**Next:** Run the dashboard and test with `user_upload_sample.csv`!

---

**Dashboard URL:** http://localhost:8503  
**Database:** salesdata @ localhost:3307  
**Master Data:** `D:\Dashboard Code\NO_WH\DS\barcode_matcher\sample_data\barcodedata.csv`
