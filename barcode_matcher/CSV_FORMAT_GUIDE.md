## 📋 CSV FORMAT GUIDE FOR BARCODE MATCHING

### ✅ CORRECT FORMAT

Your upload file must have **SINGLE COLUMN** named `barcode`:

```csv
barcode
033844004019
4607048108727
G524
5281003554102
8888056103614
```

### 🔍 Format Rules:

1. **Column Name:** Must be exactly `barcode` (lowercase)
2. **No Extra Columns:** Just one column
3. **No Headers in Data:** First row should be `barcode`, then data starts
4. **Barcode Types Supported:**
   - Numeric: `033844004019`, `4607048108727`
   - Alphanumeric: `G524`, `X715`, `A914`
   - Mixed: Any combination

### 📊 Saving Your CSV

**Excel → CSV:**
1. Open your Excel file
2. Delete all columns except barcode column
3. Rename column header to `barcode`
4. File → Save As → CSV (Comma delimited) (*.csv)
5. **Important:** Choose "CSV UTF-8 (Comma delimited)" if available

**Example Excel to CSV:**

Excel file:
| Barcode | Quantity | Location |
|---------|----------|----------|
| 033844004019 | 10 | Store A |
| 4607048108727 | 5 | Store B |

Delete extra columns → Keep only:
| barcode |
|---------|
| 033844004019 |
| 4607048108727 |

Save as CSV → Result:
```csv
barcode
033844004019
4607048108727
```

### ⚠️ Common Issues & Fixes

**Issue 1: Numbers shown as 1.23457E+12 (Scientific notation)**
- **Solution:** Format column as "Text" before saving
  1. Select barcode column
  2. Right-click → Format Cells
  3. Choose "Text"
  4. Re-type or paste barcodes
  5. Save as CSV

**Issue 2: Leading zeros removed (08800 becomes 8800)**
- **Solution:** Format as Text (same as above)
- Or prefix with apostrophe: `'08800`

**Issue 3: Numbers have .0 suffix (123.0)**
- **Solution:** System auto-removes .0 suffix, no action needed

**Issue 4: Multiple columns in file**
- **Solution:** Delete all except barcode column
- Or specify column name in dashboard

### 📝 Test Your Format

**Good CSV Example:**
```csv
barcode
033844004019
4607048108727
G524
5281003554102
```

**Bad CSV Example (will fail):**
```csv
Barcode,Quantity,Location
033844004019,10,Store A
4607048108727,5,Store B
```
❌ Reason: Multiple columns, wrong header name

### 🎯 Quick Check

Open your CSV in Notepad (not Excel):
```
barcode
033844004019
4607048108727
G524
```

Should see:
- ✅ First line: `barcode`
- ✅ Each line after: one barcode value
- ✅ No commas in data (unless part of barcode)
- ✅ No quotes around numbers

### 💡 Best Practice

1. **Keep original Excel file** (with all columns)
2. **Create separate file for upload:**
   - Copy barcode column to new sheet
   - Rename header to `barcode`
   - Save as CSV UTF-8
3. **Test with small file first** (5-10 barcodes)

### 🔧 If Numbers Still Don't Match

**Problem:** Barcodes in database have leading zeros, your file doesn't

**Example:**
- Database: `033844004019` (with leading zeros)
- Your file: `33844004019` (missing leading zero)
- Result: ❌ NO MATCH

**Solution:**
1. Format column as Text in Excel
2. Ensure leading zeros preserved
3. Or pad zeros in Excel: `=TEXT(A2,"000000000000")`

### 📞 Still Having Issues?

Check your actual barcodes:
```sql
-- Check what's in database
SELECT vc_item_barcode FROM barcode_item_master LIMIT 10;

-- Search for specific barcode
SELECT * FROM barcode_item_master 
WHERE vc_item_barcode = '033844004019';
```

Compare with your CSV file format!
