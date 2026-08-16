"""
Quick test script - verifies data fetching and feature engineering work.
Run: python test_quick.py
"""
import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 50)
print("Quick Test - Data & Features")
print("=" * 50)

# Test 1: Data Fetcher
print("\n[1] Testing data fetcher...")
try:
    from src.data_fetcher import fetch_stock_data
    df = fetch_stock_data('0700', years=1)
    print(f"  OK: {len(df)} rows fetched")
    print(f"  Columns: {list(df.columns)}")
except Exception as e:
    print(f"  FAIL: {e}")

# Test 2: Feature Engineering
print("\n[2] Testing feature engineering...")
try:
    from src.feature_engineering import compute_features, FEATURE_COLUMNS
    df_features = compute_features(df)
    print(f"  OK: {len(FEATURE_COLUMNS)} features computed")
    print(f"  Features: {FEATURE_COLUMNS}")
except Exception as e:
    print(f"  FAIL: {e}")

# Test 3: Market Context (direct yfinance)
print("\n[3] Testing market context fetch...")
try:
    import yfinance as yf
    from datetime import datetime, timedelta
    import pytz

    end_date = datetime.now(pytz.timezone('Asia/Hong_Kong'))
    start_date = end_date - timedelta(days=30)

    hsi = yf.download('^HSI', start=start_date, end=end_date, progress=False)
    if not hsi.empty:
        if isinstance(hsi.columns, pd.MultiIndex):
            hsi.columns = hsi.columns.get_level_values(0)
        print(f"  OK: HSI fetched {len(hsi)} rows")
    else:
        print(f"  WARN: HSI empty")
except Exception as e:
    print(f"  FAIL: {e}")

# Test 4: Feature count check
print("\n[4] Feature count check...")
try:
    expected = 23
    actual = len(FEATURE_COLUMNS)
    if actual == expected:
        print(f"  OK: {actual} features (expected {expected})")
    else:
        print(f"  WARN: {actual} features (expected {expected})")
except Exception as e:
    print(f"  FAIL: {e}")

# Test 5: Data shape check
print("\n[5] Data shape check...")
try:
    df_clean = df_features.dropna()
    print(f"  OK: {len(df_clean)} rows after dropna (from {len(df_features)})")
except Exception as e:
    print(f"  FAIL: {e}")

print("\n" + "=" * 50)
print("Test complete!")
print("=" * 50)
