# scripts/fix_data_structure.py
import pandas as pd
from pathlib import Path
import re

def fix_data_structure():
    """Fix the transposed/misaligned data structure"""
    
    csv_path = Path("../kb/master_data.csv")
    print("="*60)
    print(f"FIXING DATA STRUCTURE: {csv_path.resolve()}")
    print("="*60)
    
    # Read the original data
    df = pd.read_csv(csv_path, low_memory=False)
    print(f"Original shape: {df.shape}")
    
    # Let's examine the first few rows to understand the structure
    print("\nFirst 5 rows of first 5 columns:")
    print(df.iloc[:5, :5].to_string())
    
    # The issue: looks like we need to transpose or restructure
    # Let's find where the actual headers are
    
    # Check if first row contains plan names
    first_row = df.iloc[0, :].fillna('')
    print(f"\nFirst row values: {first_row[:10].tolist()}")
    
    # Check if we can find a row that looks like headers
    potential_header_rows = []
    for i in range(min(10, len(df))):
        row_vals = df.iloc[i, :10].fillna('').astype(str)
        # Look for rows with plan-like names
        plan_indicators = sum(1 for val in row_vals if any(x in val.lower() 
                             for x in ['e1', 'e3', 'e5', 'basic', 'standard', 'premium', 'business']))
        if plan_indicators >= 3:
            potential_header_rows.append((i, plan_indicators, row_vals.tolist()))
    
    print(f"\nPotential header rows found: {len(potential_header_rows)}")
    for i, count, vals in potential_header_rows:
        print(f"  Row {i} ({count} plan indicators): {vals}")
    
    # Try to identify the correct structure
    if len(potential_header_rows) > 0:
        header_row_idx = potential_header_rows[0][0]
        print(f"\nUsing row {header_row_idx} as headers")
        
        # Extract the corrected structure
        new_headers = df.iloc[header_row_idx, :].fillna(f'unnamed_{header_row_idx}').astype(str)
        feature_data = df.iloc[header_row_idx+1:, :]
        
        # Create new dataframe with proper structure
        fixed_df = pd.DataFrame(feature_data.values, columns=new_headers)
        
        # Reset index
        fixed_df = fixed_df.reset_index(drop=True)
        
        print(f"\nFixed structure shape: {fixed_df.shape}")
        print(f"New columns: {fixed_df.columns[:10].tolist()}")
        
    else:
        print("\nCannot automatically detect header row. Manual inspection needed.")
        
        # Let's try a different approach - look for the 'Feature' value
        feature_locations = []
        for i in range(min(20, len(df))):
            for j in range(min(10, len(df.columns))):
                if str(df.iloc[i, j]).lower().strip() == 'feature':
                    feature_locations.append((i, j))
        
        print(f"Found 'Feature' at locations: {feature_locations}")
        
        if feature_locations:
            feat_row, feat_col = feature_locations[0]
            print(f"Using 'Feature' location as reference: row {feat_row}, col {feat_col}")
            
            # Extract features and plans based on this reference
            if feat_col == 0:  # Features are in first column
                features = df.iloc[feat_row+1:, feat_col].dropna().astype(str)
                plan_headers = df.iloc[feat_row, feat_col+1:].dropna().astype(str)
                
                print(f"Found {len(features)} features and {len(plan_headers)} plans")
                print(f"Sample features: {features.head().tolist()}")
                print(f"Sample plans: {plan_headers.head().tolist()}")
                
                # Create the matrix
                matrix_data = df.iloc[feat_row+1:feat_row+1+len(features), feat_col+1:feat_col+1+len(plan_headers)]
                
                fixed_df = pd.DataFrame(matrix_data.values, 
                                      index=features.values,
                                      columns=plan_headers.values)
                fixed_df.index.name = 'Feature'
                
            else:
                print("Feature column is not first - need manual adjustment")
                return None
        else:
            print("Cannot find 'Feature' reference point")
            return None
    
    # Clean up the fixed dataframe
    if 'fixed_df' in locals():
        # Remove completely empty rows and columns
        fixed_df = fixed_df.dropna(how='all', axis=0).dropna(how='all', axis=1)
        
        # Clean column names
        fixed_df.columns = [clean_column_name(col) for col in fixed_df.columns]
        
        print(f"\nFinal cleaned shape: {fixed_df.shape}")
        print(f"Final columns: {fixed_df.columns[:10].tolist()}")
        
        # Show sample of fixed data
        print(f"\nSample of fixed data:")
        print(fixed_df.iloc[:5, :5].to_string())
        
        # Save the fixed data
        output_path = Path("../kb/master_data_fixed.csv")
        output_path.parent.mkdir(exist_ok=True)  # Create kb directory if it doesn't exist
        fixed_df.to_csv(output_path)
        print(f"\nSaved fixed data to: {output_path.resolve()}")
        
        return fixed_df
    
    return None

def clean_column_name(col_name):
    """Clean up column names"""
    if pd.isna(col_name):
        return f'unnamed_{hash(str(col_name)) % 1000}'
    
    col_str = str(col_name).strip()
    
    # Remove double characters (ddaattaa -> data)
    cleaned = re.sub(r'(.)\1+', r'\1', col_str)
    
    # Clean up common issues
    cleaned = cleaned.replace('_', ' ')
    cleaned = re.sub(r'\s+', ' ', cleaned)  # Multiple spaces to single
    cleaned = cleaned.title()  # Proper case
    
    return cleaned

def analyze_fixed_data(df):
    """Analyze the fixed data structure"""
    if df is None:
        return
        
    print("\n" + "="*60)
    print("ANALYSIS OF FIXED DATA")
    print("="*60)
    
    print(f"Shape: {df.shape[0]} features × {df.shape[1]} plans")
    
    # Analyze unique values in the matrix
    all_values = set()
    for col in df.columns:
        all_values.update(df[col].dropna().unique())
    
    print(f"Unique values in matrix: {sorted(all_values)}")
    
    # Count feature availability
    checkmark_counts = {}
    for col in df.columns:
        checkmarks = (df[col] == '✔').sum()
        checkmark_counts[col] = checkmarks
    
    print(f"\nFeature counts per plan:")
    for plan, count in sorted(checkmark_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {plan}: {count} features")
    
    # Sample features
    print(f"\nSample features:")
    feature_names = df.index[:10].tolist() if hasattr(df, 'index') else []
    for i, feat in enumerate(feature_names, 1):
        print(f"  {i}. {feat}")

if __name__ == "__main__":
    fixed_df = fix_data_structure()
    if fixed_df is not None:
        analyze_fixed_data(fixed_df)
        print("\n" + "="*60)
        print("✅ DATA STRUCTURE FIXED!")
        print("Next step: Proceed to Neo4j setup with the fixed data")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("❌ COULD NOT FIX DATA STRUCTURE AUTOMATICALLY")
        print("Manual inspection and fixing required")
        print("="*60)