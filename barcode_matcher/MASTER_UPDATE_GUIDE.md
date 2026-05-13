# 📘 Master Table Update Guide

## Overview
The barcode matcher now has **TWO ways** to update the `barcode_item_master` table.

---

## ✅ NEW FEATURE: Save Upload with Master Update

### How It Works

When you upload a file with barcode-to-itemcode mappings:

```
Upload Flow:
┌─────────────────────────────────────────────────────────┐
│ 1. Upload CSV/Excel with columns:                       │
│    - barcode: The barcode value                         │
│    - itemcode: The corresponding item code              │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│ 2. App processes and shows results                      │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│ 3. Check: ➕ "Also add NEW barcode mappings"           │
│    to Master Table                                      │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│ 4. Click: "💾 Save Upload"                             │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│ UPDATES:                                                │
│ ✅ barcode_upload_history (always)                     │
│ ✅ barcode_upload_details (always)                     │
│ ✅ barcode_item_master (ONLY if checkbox checked)      │
└─────────────────────────────────────────────────────────┘
```

---

## 🎯 What Gets Added to Master Table

**Only if checkbox is checked:**

1. ✅ **Matched rows** (barcode + item code found)
2. ✅ **NEW barcodes** (not already in master table)
3. ❌ **Skips existing** barcodes (no duplicates)
4. ❌ **Skips unmatched** rows (no item code)

### Example:

Your upload has 100 rows:
- 80 Matched
- 20 Not Found

Master table has 229,017 barcodes.

Of the 80 matched:
- 75 already exist in master → **Skipped**
- 5 are NEW → **Added to master** ✅

**Result:** 5 new mappings added to `barcode_item_master`

---

## 📊 Two Methods Compared

| Feature | Tab 1: Save Upload | Tab 4: Run Daily Load |
|---------|-------------------|----------------------|
| **Updates Master?** | ✅ Yes (if checkbox checked) | ✅ Yes (always) |
| **Source** | Your uploaded file | barcodedata.csv |
| **When to use** | Add specific new mappings | Bulk daily refresh |
| **Saves history?** | ✅ Yes | ❌ No |
| **Tracks unmatched?** | ✅ Yes | ❌ No |
| **User control** | ✅ Checkbox option | Automatic |

---

## 🔄 Recommended Workflow

### For Daily Operations:
**Tab 4: "Run Daily Load"**
- Updates master from official barcodedata.csv
- Adds all new barcodes from source file
- Run once per day

### For Ad-hoc Additions:
**Tab 1: "Save Upload" with checkbox**
- Upload small file with new barcode-itemcode pairs
- Check "Also add NEW barcode mappings"
- Click "Save Upload"
- New mappings instantly available for search

---

## 🧪 Testing the Feature

### Test File Format:
```csv
barcode,itemcode
123456789012,ITEM001
987654321098,ITEM002
```

Or with just barcode column (if items exist in another source):
```csv
barcode
070177109868
070177067762
```

### Steps:
1. Upload test file
2. See match results
3. ✅ Check "Also add NEW barcode mappings to Master Table"
4. Click "💾 Save Upload"
5. Look for message: "➕ Added X new barcode mappings to master table!"
6. Test search in Tab 2 to verify new barcodes are searchable

---

## ⚠️ Important Notes

1. **Uniqueness:** Barcode column has UNIQUE constraint
   - Same barcode cannot exist twice
   - `ON CONFLICT DO NOTHING` prevents errors

2. **Validation:** App checks before inserting
   - Only inserts if NOT already in master
   - Safe to upload same file multiple times

3. **Cache:** After adding, cache is cleared
   - Next search will use updated master data
   - Immediate availability

4. **Audit Trail:** All uploads tracked in history
   - Tab 3 shows all past uploads
   - Can review what was added when

---

## 🐛 Troubleshooting

**"0 new mappings added" but I expected some:**
- Check if barcodes already exist in master (Tab 2: Search)
- Verify item codes are present (not blank)
- Check match_status = "Matched"

**"Added but can't find in search":**
- Wait 2 seconds for cache to clear
- Refresh browser page
- Check exact barcode format (leading zeros)

**"Error saving to database":**
- Check PostgreSQL is running (port 3307)
- Verify table exists: `barcode_item_master`
- Check database connection in config

---

## 📈 Monitoring

After saving with master update:

```sql
-- Check total barcodes
SELECT COUNT(*) FROM barcode_item_master WHERE is_active = TRUE;

-- Check recent additions
SELECT vc_item_barcode, vc_item_code, created_at
FROM barcode_item_master
WHERE created_at > NOW() - INTERVAL '1 hour'
ORDER BY created_at DESC;

-- Check specific barcode
SELECT * FROM barcode_item_master 
WHERE vc_item_barcode = '070177109868';
```

---

## 🎉 Benefits

1. **Flexibility:** Choose when to update master
2. **Safety:** Checkbox prevents accidental updates
3. **Visibility:** Shows count of new mappings added
4. **Tracking:** All changes logged in history
5. **Speed:** Immediate availability in search
