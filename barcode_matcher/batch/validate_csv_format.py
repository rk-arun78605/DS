'''
CSV Format Validator for Barcode Upload
Checks if your CSV is in correct format before uploading
'''
import pandas as pd
import sys

def validate_csv(filepath):
    '''Validate CSV format for barcode matching'''
    print("=" * 60)
    print("BARCODE CSV FORMAT VALIDATOR")
    print("=" * 60)
    print(f"\nChecking file: {filepath}\n")
    
    errors = []
    warnings = []
    
    try:
        # Try reading with different encodings
        df = None
        for encoding in ['utf-8', 'latin1', 'cp1252', 'iso-8859-1']:
            try:
                df = pd.read_csv(filepath, encoding=encoding)
                print(f"✅ File readable with encoding: {encoding}")
                break
            except:
                continue
        
        if df is None:
            print("❌ FAILED: Cannot read file with any encoding")
            return False
        
        # Check 1: Column count
        if len(df.columns) > 1:
            warnings.append(f"Multiple columns found: {df.columns.tolist()}")
            warnings.append("System will only use 'barcode' column")
        else:
            print("✅ Single column detected")
        
        # Check 2: Column name
        if 'barcode' in df.columns:
            print("✅ Column named 'barcode' found")
        else:
            errors.append(f"Column 'barcode' not found. Available: {df.columns.tolist()}")
            errors.append("Rename your column to 'barcode' (lowercase)")
        
        if errors:
            print("\n❌ ERRORS FOUND:")
            for i, err in enumerate(errors, 1):
                print(f"   {i}. {err}")
            return False
        
        # Get barcode column
        barcode_col = 'barcode' if 'barcode' in df.columns else df.columns[0]
        
        # Check 3: Empty values
        empty_count = df[barcode_col].isna().sum()
        if empty_count > 0:
            warnings.append(f"{empty_count} empty barcode values found (will be skipped)")
        
        # Check 4: Format analysis
        df['barcode_str'] = df[barcode_col].astype(str)
        
        # Trailing zeros check (these are REAL, not corruption!)
        trailing_zero_barcodes = df[df['barcode_str'].str.match(r'^[0-9]+0{3,}$', na=False)]
        if not trailing_zero_barcodes.empty:
            print(f"ℹ️  {len(trailing_zero_barcodes)} barcodes end with multiple zeros (this is NORMAL for EAN-13/UPC)")
            print(f"   Examples: {trailing_zero_barcodes['barcode_str'].head(3).tolist()}")
        
        # Leading zeros check
        numeric_barcodes = df[df['barcode_str'].str.match(r'^[0-9]+$', na=False)]
        if not numeric_barcodes.empty:
            leading_zero_count = numeric_barcodes[numeric_barcodes['barcode_str'].str.match(r'^0')].shape[0]
            if leading_zero_count > 0:
                print(f"✅ {leading_zero_count} barcodes with leading zeros preserved")
            
            # Check for scientific notation indicators
            if (numeric_barcodes['barcode_str'].str.len() > 12).any():
                warnings.append("Some barcodes are very long (>12 digits) - verify Excel didn't use scientific notation")
        
        # Check 5: Sample preview
        print(f"\n📊 File Statistics:")
        print(f"   Total rows: {len(df)}")
        print(f"   Non-empty barcodes: {len(df) - empty_count}")
        
        # Show samples
        print(f"\n📋 First 10 barcodes:")
        sample_data = df[barcode_col].head(10)
        for idx, barcode in enumerate(sample_data, 1):
            barcode_str = str(barcode) if pd.notna(barcode) else '(empty)'
            print(f"   {idx:2d}. {barcode_str}")
        
        # Check 6: Potential issues
        has_spaces = df['barcode_str'].str.contains(' ', na=False).any()
        if has_spaces:
            warnings.append("Some barcodes contain spaces (will be auto-trimmed)")
        
        has_decimal = df['barcode_str'].str.contains(r'\.', na=False).any()
        if has_decimal:
            warnings.append("Some barcodes have decimal points (e.g., 123.0 → will be cleaned to 123)")
        
        # Show warnings
        if warnings:
            print("\n⚠️  WARNINGS:")
            for i, warn in enumerate(warnings, 1):
                print(f"   {i}. {warn}")
        
        print("\n" + "=" * 60)
        print("✅ FILE FORMAT IS VALID FOR UPLOAD!")
        print("=" * 60)
        print("\nYou can upload this file to the dashboard.")
        print("Dashboard URL: http://localhost:8503")
        
        return True
    
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        return False

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python validate_csv_format.py <path_to_csv_file>")
        print("\nExample:")
        print('  python validate_csv_format.py "C:\\Users\\Downloads\\mapitemcode.csv"')
        sys.exit(1)
    
    filepath = sys.argv[1]
    validate_csv(filepath)
