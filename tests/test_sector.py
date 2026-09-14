"""Tests for sector rotation module."""
import os
import sys
import pandas as pd
import numpy as np
import pytest
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestSector:
    """Tests for sector rotation features."""
    
    def test_import(self):
        from src.sector import compute_sector_features
        assert callable(compute_sector_features)
    
    def test_compute_sector_features(self):
        from src.sector import compute_sector_features
        
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
        
        result = compute_sector_features(df, '0700')
        
        assert 'sector_momentum_5d' in result.columns
        assert 'sector_momentum_20d' in result.columns
        assert 'sector_vs_hsi' in result.columns
        assert len(result) == 100
    
    def test_sector_features_fill_na(self):
        from src.sector import compute_sector_features
        
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
        
        result = compute_sector_features(df, '0700')
        
        assert not result['sector_momentum_5d'].isna().any()
        assert not result['sector_momentum_20d'].isna().any()
        assert not result['sector_vs_hsi'].isna().any()
    
    def test_sector_values_range(self):
        from src.sector import compute_sector_features
        
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
        
        result = compute_sector_features(df, '0700')
        
        assert result['sector_momentum_5d'].between(-1, 1).all()
        assert result['sector_momentum_20d'].between(-1, 1).all()
        assert result['sector_vs_hsi'].between(-1, 1).all()
