"""Tests for sentiment module."""
import os
import sys
import pandas as pd
import numpy as np
import pytest
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestSentiment:
    """Tests for news sentiment features."""
    
    def test_import(self):
        from src.sentiment import compute_sentiment_features
        assert callable(compute_sentiment_features)
    
    def test_compute_sentiment_features_no_news(self):
        from src.sentiment import compute_sentiment_features
        
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
        
        result = compute_sentiment_features(df, '0700')
        
        assert 'sentiment_5d' in result.columns
        assert 'sentiment_10d' in result.columns
        assert 'sentiment_change' in result.columns
        assert len(result) == 100
    
    def test_sentiment_features_fill_na(self):
        from src.sentiment import compute_sentiment_features
        
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
        
        result = compute_sentiment_features(df, '0700')
        
        assert not result['sentiment_5d'].isna().any()
        assert not result['sentiment_10d'].isna().any()
        assert not result['sentiment_change'].isna().any()
    
    def test_sentiment_values_range(self):
        from src.sentiment import compute_sentiment_features
        
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
        
        result = compute_sentiment_features(df, '0700')
        
        assert result['sentiment_5d'].between(-1, 1).all()
        assert result['sentiment_10d'].between(-1, 1).all()
        assert result['sentiment_change'].between(-2, 2).all()
