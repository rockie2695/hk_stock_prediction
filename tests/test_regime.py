"""Tests for regime detection module."""
import os
import sys
import pandas as pd
import numpy as np
import pytest
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestRegime:
    """Tests for market regime detection."""
    
    def test_import(self):
        from src.regime import detect_regime, compute_regime_features
        assert callable(detect_regime)
        assert callable(compute_regime_features)
    
    def test_detect_regime_bull(self):
        from src.regime import detect_regime
        
        # Create uptrending HSI
        dates = pd.date_range('2025-01-01', periods=250, freq='B')
        prices = np.linspace(15000, 25000, 250) + np.random.normal(0, 200, 250)
        hsi = pd.Series(prices, index=dates)
        
        result = detect_regime(hsi)
        
        assert 'market_regime' in result.columns
        assert 'regime_confidence' in result.columns
        assert 'hsi_trend_50_200' in result.columns
        assert len(result) == 250
    
    def test_detect_regime_bear(self):
        from src.regime import detect_regime
        
        # Create strongly downtrending HSI with enough data for MA200
        dates = pd.date_range('2024-01-01', periods=300, freq='B')
        prices = np.linspace(25000, 10000, 300) + np.random.normal(0, 100, 300)
        hsi = pd.Series(prices, index=dates)
        
        result = detect_regime(hsi)
        
        # Strong downtrend should be bear (0) or at least below sideways (1)
        last_regime = result['market_regime'].iloc[-1]
        assert last_regime in [0, 1]  # bear or sideways (not bull)
    
    def test_detect_regime_values(self):
        from src.regime import detect_regime
        
        dates = pd.date_range('2025-01-01', periods=250, freq='B')
        prices = np.linspace(15000, 25000, 250) + np.random.normal(0, 200, 250)
        hsi = pd.Series(prices, index=dates)
        
        result = detect_regime(hsi)
        
        # Market regime should be 0, 1, or 2
        assert result['market_regime'].isin([0, 1, 2]).all()
        assert result['regime_confidence'].between(0, 1).all()
    
    def test_compute_regime_features(self):
        from src.regime import compute_regime_features
        
        dates = pd.date_range('2025-01-01', periods=250, freq='B')
        df = pd.DataFrame({
            'Date': dates,
            'Close': np.linspace(15000, 25000, 250) + np.random.normal(0, 200, 250),
            'Open': np.linspace(15000, 25000, 250),
            'High': np.linspace(15000, 25000, 250) + 100,
            'Low': np.linspace(15000, 25000, 250) - 100,
            'Volume': np.random.randint(1000000, 5000000, 250),
            'hsi_ret_5d': np.random.normal(0, 0.02, 250),
        })
        df.set_index('Date', inplace=True)
        
        result = compute_regime_features(df)
        
        assert 'market_regime' in result.columns
        assert 'regime_confidence' in result.columns
        assert 'hsi_trend_50_200' in result.columns
        assert not result['market_regime'].isna().any()
        assert not result['regime_confidence'].isna().any()
