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

    def test_fetch_sentiment_returns_articles(self):
        """Verify fetch_sentiment returns non-zero articles with non-zero scores.
        驗證 fetch_sentiment 返回非零文章和非零分數。"""
        from src.sentiment import fetch_sentiment
        
        # Clear cache to force fresh fetch / 清除快取以強制重新獲取
        cache_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'cache', '0700_sentiment.parquet'
        )
        if os.path.exists(cache_path):
            os.remove(cache_path)
        
        df = fetch_sentiment('0700', days=30)
        
        assert not df.empty, "Sentiment should return articles / 情緒應返回文章"
        assert len(df) > 0, "Should have at least 1 article / 至少應有1篇文章"
        assert 'sentiment_score' in df.columns
        assert 'date' in df.columns
        # Verify scores are not all zero (news has bullish/bearish keywords)
        # 驗證分數不全為零 (新聞包含看多/看空關鍵字)
        assert (df['sentiment_score'] != 0).any(), \
            "Sentiment scores should have non-zero values from news keywords / 情緒分數應有非零值"

    def test_fetch_sentiment_date_parsing(self):
        """Verify date column is parsed correctly from AKShare '发布时间'.
        驗證日期欄位從 AKShare '发布时间' 正確解析。"""
        from src.sentiment import fetch_sentiment
        
        df = fetch_sentiment('0700', days=30)
        
        if not df.empty:
            assert pd.api.types.is_datetime64_any_dtype(df['date']), \
                "Date column should be datetime type / 日期欄位應為 datetime 類型"
