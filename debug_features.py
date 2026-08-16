"""
Debug script - check feature values during prediction
"""
import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_fetcher import fetch_stock_data
from src.feature_engineering import compute_features, FEATURE_COLUMNS

print("=" * 60)
print("Debug: Feature Values Check")
print("=" * 60)

# Expected features (same as in feature_engineering.py)
expected_features = FEATURE_COLUMNS.copy()
print(f"\nExpected features ({len(expected_features)}): {expected_features}")

# Fetch stock data
print("\nFetching stock data for 0700...")
df = fetch_stock_data('0700', years=1)
print(f"Raw data: {len(df)} rows")

# Compute features
df = compute_features(df)
print(f"After features: {len(df)} rows")

# Check for market context features
print("\n--- Market context features ---")
for col in ['hsi_ret_5d', 'hsi_ret_20d', 'usdhkd_change']:
    if col in df.columns:
        print(f"  {col}: {df[col].isna().sum()} NaN out of {len(df)}")
        if df[col].notna().any():
            print(f"    Last value: {df[col].iloc[-1]:.6f}")
    else:
        print(f"  {col}: NOT FOUND in dataframe")

# Check last row values
print("\n--- Last row feature values ---")
for col in expected_features:
    if col in df.columns:
        val = df[col].iloc[-1]
        print(f"  {col}: {val} (NaN: {pd.isna(val)})")
    else:
        print(f"  {col}: MISSING")

# Check feature stats
print("\n--- Feature statistics (last 20 rows) ---")
last_20 = df[expected_features].tail(20)
for col in expected_features:
    if col in last_20.columns:
        vals = last_20[col]
        print(f"  {col}: min={vals.min():.4f}, max={vals.max():.4f}, std={vals.std():.4f}")

print("\n" + "=" * 60)
print("Debug complete!")
print("=" * 60)
