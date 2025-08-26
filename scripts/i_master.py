# scripts/explore_data_structure.py
import pandas as pd
from pathlib import Path
import re

def clean_column_name(col_name):
    """Clean up garbled column names"""
    if pd.isna(col_name):
        return col_name
    
    # Remove double characters (ddaattaa -> data)
    cleaned = re.sub(r'(.)\1+', r'\1', str(col_name))
    
    # Replace underscores with spaces for readability
    cleaned = cleaned.replace('_', ' ')
    
    return cleaned

def explore_data_structure():
    csv_path = Path(r"D:\m365\kb\master_data.csv")
    print("="*60)
    print(f"EXPLORING: {csv_path.resolve()}")
    print("="*60)
    
    df = pd.read_csv(csv_path, low_memory=False)
    print(f"Dataset shape: {df.shape[0]:,} rows × {df.shape[1]} columns\n")
    
    # 1. Examine the first column (likely contains feature names)
    print("1. FIRST COLUMN ANALYSIS (Likely Features)")
    print("-" * 40)
    first_col = df.columns[0]
    print(f"Column name: '{first_col}'")
    print(f"Non-null values: {df.iloc[:, 0].count()}")
    print(f"Unique values: {df.iloc[:, 0].nunique()}")
    
    print("\nFirst 15 values in first column:")
    for i, val in enumerate(df.iloc[:15, 0], 1):
        print(f"{i:2d}. {val}")
    
    print("\n" + "="*60)
    
    # 2. Look for Microsoft 365 plan columns
    print("2. MICROSOFT 365 PLAN COLUMNS")
    print("-" * 40)
    
    plan_keywords = ['e1', 'e3', 'e5', 'business', 'enterprise', 'frontline', 'education', 'f1', 'f3']
    plan_cols = []
    
    for col in df.columns:
        col_lower = str(col).lower()
        if any(keyword in col_lower for keyword in plan_keywords):
            cleaned_name = clean_column_name(col)
            plan_cols.append((col, cleaned_name))
    
    print(f"Found {len(plan_cols)} potential plan columns:")
    for orig, cleaned in plan_cols[:15]:  # Show first 15
        print(f"  '{orig}' -> '{cleaned}'")
    
    if len(plan_cols) > 15:
        print(f"  ... and {len(plan_cols) - 15} more")
    
    print("\n" + "="*60)
    
    # 3. Analyze data values in the matrix
    print("3. DATA VALUES ANALYSIS")
    print("-" * 40)
    
    if len(plan_cols) > 0:
        # Sample a few plan columns to see what values they contain
        sample_plans = [col[0] for col in plan_cols[:3]]
        sample_data = df[sample_plans].head(10)
        
        print("Sample data matrix (first 10 rows, first 3 plan columns):")
        print(sample_data.to_string())
        
        # Check unique values across these columns
        print(f"\nUnique values in plan columns:")
        all_plan_values = set()
        for col in sample_plans:
            unique_vals = df[col].dropna().unique()
            all_plan_values.update(unique_vals)
        
        print(f"Found these unique values: {sorted(all_plan_values)}")
        
    print("\n" + "="*60)
    
    # 4. Check for text-rich columns
    print("4. TEXT-RICH COLUMNS ANALYSIS")
    print("-" * 40)
    
    text_keywords = ['desc', 'description', 'title', 'body', 'content', 'details', 'summary', 'text', 'name']
    text_cols = []
    
    for col in df.columns:
        col_lower = str(col).lower()
        if any(keyword in col_lower for keyword in text_keywords):
            text_cols.append(col)
    
    print(f"Found {len(text_cols)} potential text columns: {text_cols}")
    
    for col in text_cols:
        print(f"\nColumn: '{col}'")
        non_null = df[col].dropna()
        print(f"  Non-null values: {len(non_null)}")
        if len(non_null) > 0:
            print(f"  Sample values:")
            for i, val in enumerate(non_null.head(5), 1):
                val_str = str(val)[:100] + "..." if len(str(val)) > 100 else str(val)
                print(f"    {i}. {val_str}")
    
    print("\n" + "="*60)
    
    # 5. Metadata columns
    print("5. METADATA COLUMNS")
    print("-" * 40)
    
    metadata_keywords = ['source', 'file', 'hash', 'processed', 'extraction', 'thread', 'row_id']
    metadata_cols = []
    
    for col in df.columns:
        col_lower = str(col).lower()
        if any(keyword in col_lower for keyword in metadata_keywords):
            metadata_cols.append(col)
    
    print(f"Found {len(metadata_cols)} metadata columns:")
    for col in metadata_cols:
        unique_count = df[col].nunique()
        print(f"  '{col}' - {unique_count} unique values")
        if unique_count < 10:  # Show sample if few unique values
            print(f"    Sample: {list(df[col].dropna().unique()[:5])}")
    
    print("\n" + "="*60)
    
    # 6. Feature extraction suggestions
    print("6. FEATURE EXTRACTION SUGGESTIONS")
    print("-" * 40)
    
    feature_col = df.columns[0]  # Assuming first column has features
    sample_features = df[feature_col].dropna().head(10).tolist()
    
    print("Based on analysis, here's what we can extract:")
    print(f"✓ Feature Names: From column '{feature_col}' ({df[feature_col].count()} features)")
    print(f"✓ Plan Names: {len(plan_cols)} different M365 plans")
    print(f"✓ Feature-Plan Matrix: Availability indicators (✔, NaN, etc.)")
    
    if text_cols:
        print(f"✓ Additional Text: {len(text_cols)} text columns with descriptions")
    
    print(f"\nNext steps:")
    print("1. Clean and normalize feature names")
    print("2. Clean and normalize plan names")  
    print("3. Create feature-plan availability mappings")
    print("4. Generate text descriptions for RAG system")
    print("5. Structure data for Neo4j graph database")
    
    return {
        'df': df,
        'plan_cols': plan_cols,
        'text_cols': text_cols,
        'metadata_cols': metadata_cols,
        'feature_col': feature_col
    }

if __name__ == "__main__":
    result = explore_data_structure()
    print("\n" + "="*60)
    print("EXPLORATION COMPLETE!")
    print("="*60)