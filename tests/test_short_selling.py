"""Tests for short selling module."""
import os
import sys
import pandas as pd
import numpy as np
import pytest
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestShortSelling:
    """Tests for short selling features."""
    
    def test_import(self):
        from src.short_selling import compute_short_selling_features
        assert callable(compute_short_selling_features)
    
    def test_compute_short_selling_features(self):
        from src.short_selling import compute_short_selling_features
        
        dates = pd.date_range('2026-01-01', periods=100, freq='B')
        df = pd.DataFrame({
            'Date': dates,
            'Close': np.random.uniform(100, 200, 100),
            'Open': np.random.uniform(100, 200, 100),
            'High': np.random.uniform(100, 200, 100),
            'Low': np.random.uniform(100, 200, 100),
            'Volume': np.random.randint(1000000, 5000000, 100),
        })
        df.set_index('Date', inplace=True)
        
        result = compute_short_selling_features(df, '0700')
        
        assert 'short_sell_ratio' in result.columns
        assert 'short_sell_ratio_5d' in result.columns
        assert 'short_sell_ratio_change' in result.columns
        assert len(result) == 100
    
    def test_short_selling_values_range(self):
        from src.short_selling import compute_short_selling_features
        
        dates = pd.date_range('2026-01-01', periods=100, freq='B')
        df = pd.DataFrame({
            'Date': dates,
            'Close': np.random.uniform(100, 200, 100),
            'Open': np.random.uniform(100, 200, 100),
            'High': np.random.uniform(100, 200, 100),
            'Low': np.random.uniform(100, 200, 100),
            'Volume': np.random.randint(1000000, 5000000, 100),
        })
        df.set_index('Date', inplace=True)
        
        result = compute_short_selling_features(df, '0700')
        
        assert result['short_sell_ratio'].between(0, 1).all()
        assert result['short_sell_ratio_5d'].between(0, 1).all()
        assert result['short_sell_ratio_change'].between(-1, 1).all()
    
    def test_short_selling_fill_na(self):
        from src.short_selling import compute_short_selling_features
        
        dates = pd.date_range('2026-01-01', periods=50, freq='B')
        df = pd.DataFrame({
            'Date': dates,
            'Close': np.random.uniform(100, 200, 50),
            'Open': np.random.uniform(100, 200, 50),
            'High': np.random.uniform(100, 200, 50),
            'Low': np.random.uniform(100, 200, 50),
            'Volume': np.random.randint(1000000, 5000000, 50),
        })
        df.set_index('Date', inplace=True)
        
        result = compute_short_selling_features(df, '0700')
        
        assert not result['short_sell_ratio'].isna().any()
        assert not result['short_sell_ratio_5d'].isna().any()
        assert not result['short_sell_ratio_change'].isna().any()
