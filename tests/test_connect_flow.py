"""Tests for connect flow module."""
import os
import sys
import pandas as pd
import numpy as np
import pytest
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestConnectFlow:
    """Tests for northbound/southbound flow features."""
    
    def test_import(self):
        from src.connect_flow import compute_connect_features
        assert callable(compute_connect_features)
    
    def test_compute_connect_features(self):
        from src.connect_flow import compute_connect_features
        
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
        
        result = compute_connect_features(df)
        
        assert 'southbound_net_5d' in result.columns
        assert 'southbound_momentum' in result.columns
        assert 'connect_sentiment' in result.columns
        assert len(result) == 100
    
    def test_connect_features_fill_na(self):
        from src.connect_flow import compute_connect_features
        
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
        
        result = compute_connect_features(df)
        
        assert not result['southbound_net_5d'].isna().any()
        assert not result['southbound_momentum'].isna().any()
        assert not result['connect_sentiment'].isna().any()
    
    def test_connect_values_range(self):
        from src.connect_flow import compute_connect_features
        
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
        
        result = compute_connect_features(df)
        
        assert result['southbound_net_5d'].between(-1, 1).all()
        assert result['southbound_momentum'].between(-1, 1).all()
        assert result['connect_sentiment'].between(-1, 1).all()
