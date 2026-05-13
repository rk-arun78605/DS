# 🎯 BARCODE FORMAT - YOUR EXACT SETUP

## ✅ Database Format (What's Stored)

Your database has these barcode formats:
```
033844004019  ← Leading zeros PRESERVED
4607048108727 ← Numeric
G524          ← Alphanumeric
90087493      ← Numeric (no leading zeros)
087000006935  ← Leading zeros PRESERVED
X715          ← Alphanumeric
```

## 📝 YOUR UPLOAD FILE FORMAT

### ✅ CORRECT Format (Single Column CSV):

**File name:** `mapitemcode.csv` or any `.csv`

**Content:**
```csv
barcode
033844004019
4607048108727
G524
90087493
087000006935
X715
```

### 🔧 How to Create This from Excel:

**Step 1: Prepare Excel**
1. Open your Excel file with barcodes
2. Select the barcode column
3. Right-click → Format Cells → Text (⚠️ IMPORTANT!)
4. Delete all other columns (keep only barcode)
5. Rename header to lowercase: `barcode`

**Step 2: Verify Leading Zeros**
- Check: `087000006935` should NOT become `87000006935`
- Check: `033844004019` should NOT become `33844004019`
- If zeros missing: Re-format as Text and re-type values

**Step 3: Save as CSV**
1. File → Save As
2. Choose: **CSV UTF-8 (Comma delimited) (*.csv)**
3. Save to desktop or downloads folder

**Step 4: Verify in Notepad**
- Open CSV in Notepad (not Excel!)
- Should see:
```
barcode
033844004019
4607048108727
G524
```

## ⚠️ Common Excel Issues

### Issue 1: Scientific Notation
**Problem:** Excel shows `4.60705E+12` instead of `4607048108727`

**Fix:**
1. Select column
2. Format Cells → Text
3. Re-type or paste values
4. Save as CSV UTF-8

### Issue 2: Leading Zeros Removed
**Problem:** `033844004019` becomes `33844004019`

**Fix:**
1. Format column as Text FIRST
2. Then paste/type barcodes
3. Verify zeros are there
4. Save as CSV UTF-8

### Issue 3: Decimal Point Added
**Problem:** `123` becomes `123.0` in CSV

**Fix:**
- System automatically removes `.0` suffix
- No action needed!

## 🧪 Test Upload File

I've created a test file for you at:
`D:\Dashboard Code\NO_WH\DS\barcode_matcher\sample_data\test_upload_numeric.csv`

**Contents:**
```csv
barcode
33844004019
4607048108727
5281003554102
8888056103614
G524
90087493
INVALID_TEST_123
```

**Expected Results:**
- `4607048108727` → ✅ Matched (J163)
- `G524` → ✅ Matched (G524)
- `90087493` → ✅ Matched (09124)
- `33844004019` → ❌ Not Found (missing leading zero!)
- `INVALID_TEST_123` → ❌ Not Found (not in database)

## 📊 Matching Rules

**Exact Match Required:**
- Database: `033844004019`
- Your file: `033844004019` ✅ Match
- Your file: `33844004019` ❌ No Match (missing leading 0)

**Case Sensitive:**
- Database: `G524`
- Your file: `G524` ✅ Match
- Your file: `g524` ❌ No Match

**No Spaces:**
- Database: `G524`
- Your file: `G524 ` ❌ No Match (trailing space)
- System removes spaces automatically

## 🎯 Your Exact Steps

1. **Open your Excel file**
2. **Format barcode column as Text:**
   - Select entire column
   - Right-click → Format Cells → Text → OK
3. **Delete all columns except barcode**
4. **Rename header to:** `barcode` (lowercase)
5. **Save As:**
   - File type: CSV UTF-8 (Comma delimited)
   - Location: Downloads folder
6. **Upload to dashboard:** http://localhost:8503
7. **Download results**

## 📥 What You'll Get Back

**Upload:**
```csv
barcode
033844004019
4607048108727
INVALID123
G524
```

**Download:**
```csv
barcode,item_code,match_status
033844004019,D446,Matched
4607048108727,J163,Matched
INVALID123,,Not Found
G524,G524,Matched
```

✅ ALL rows preserved (matched + unmatched)  
✅ Blank item_code for non-matches  
✅ match_status shows which matched

## 🔍 Troubleshooting

**None of my barcodes match:**
- Check: Leading zeros preserved?
- Check: Column named exactly `barcode`?
- Check: Saved as CSV UTF-8?
- Check: No extra columns?

**Some match, some don't:**
- Compare your barcode with database format
- Check: `sample_db_format.csv` to see exact formats
- Leading zeros must match exactly

**Numbers appear wrong in Excel:**
- Open CSV in Notepad to see true format
- Excel may display incorrectly but CSV is correct

## 💡 Pro Tip

Keep two files:
1. **Master Excel** - All columns, all data
2. **Upload CSV** - Single `barcode` column, Text format

Create upload CSV from master when needed!
