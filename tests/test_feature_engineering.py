"""
Tests for feature_engineering.py - feature computation and technical indicators.
"""
import os
import sys
import pytest
import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.feature_engineering import (
    compute_features,
    compute_target_days,
    _compute_rsi,
    _compute_macd,
    _compute_atr,
    _compute_adx,
    _compute_stochastic,
    _compute_mfi,
    _compute_williams_r,
    filter_correlated_features,
    FEATURE_COLUMNS
)


@pytest.fixture
def sample_stock_data():
    """Create sample stock data for testing."""
    np.random.seed(42)
    n = 100
    
    # Generate realistic stock data
    dates = pd.date_range('2024-01-01', periods=n, freq='B')
    base_price = 100
    
    # Random walk for prices
    returns = np.random.normal(0.001, 0.02, n)
    prices = base_price * np.cumprod(1 + returns)
    
    df = pd.DataFrame({
        'Date': dates,
        'Open': prices * (1 + np.random.uniform(-0.01, 0.01, n)),
        'High': prices * (1 + np.random.uniform(0, 0.03, n)),
        'Low': prices * (1 - np.random.uniform(0, 0.03, n)),
        'Close': prices,
        'Volume': np.random.randint(1000000, 5000000, n)
    })
    
    # Ensure High >= Open, Close and Low <= Open, Close
    df['High'] = df[['Open', 'High', 'Close']].max(axis=1) * 1.001
    df['Low'] = df[['Open', 'Low', 'Close']].min(axis=1) * 0.999
    
    return df


class TestFeatureEngineering:
    """Test feature engineering functions."""
    
    def test_compute_features_returns_dataframe(self, sample_stock_data):
        """Test that compute_features returns a DataFrame."""
        result = compute_features(sample_stock_data)
        assert isinstance(result, pd.DataFrame)
    
    def test_compute_features_preserves_original_columns(self, sample_stock_data):
        """Test that original columns are preserved."""
        result = compute_features(sample_stock_data)
        for col in ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']:
            assert col in result.columns
    
    def test_compute_features_adds_new_columns(self, sample_stock_data):
        """Test that new feature columns are added."""
        result = compute_features(sample_stock_data)
        # Should have more columns than original
        assert len(result.columns) > len(sample_stock_data.columns)
    
    def test_compute_features_has_all_feature_columns(self, sample_stock_data):
        """Test that all FEATURE_COLUMNS are present."""
        result = compute_features(sample_stock_data)
        for col in FEATURE_COLUMNS:
            if col not in ['hsi_ret_5d', 'hsi_ret_20d', 'usdhkd_change']:  # Market features
                assert col in result.columns, f"Missing feature: {col}"
    
    def test_rsi_computation(self, sample_stock_data):
        """Test RSI computation."""
        rsi = _compute_rsi(sample_stock_data['Close'], period=14)
        assert isinstance(rsi, pd.Series)
        assert len(rsi) == len(sample_stock_data)
        # RSI should be between 0 and 100
        valid_rsi = rsi.dropna()
        assert (valid_rsi >= 0).all()
        assert (valid_rsi <= 100).all()
    
    def test_macd_computation(self, sample_stock_data):
        """Test MACD computation."""
        macd, signal, hist = _compute_macd(sample_stock_data['Close'])
        assert isinstance(macd, pd.Series)
        assert isinstance(signal, pd.Series)
        assert isinstance(hist, pd.Series)
        assert len(macd) == len(sample_stock_data)
    
    def test_bollinger_computation(self, sample_stock_data):
        """Test Bollinger Bands computation (via compute_features)."""
        result = compute_features(sample_stock_data)
        # Check that bb_width feature exists
        assert 'bb_width' in result.columns
        # bb_width should be positive
        valid_bb = result['bb_width'].dropna()
        assert (valid_bb >= 0).all()
    
    def test_atr_computation(self, sample_stock_data):
        """Test ATR computation."""
        atr = _compute_atr(sample_stock_data, period=14)
        assert isinstance(atr, pd.Series)
        assert len(atr) == len(sample_stock_data)
        # ATR should be positive
        valid_atr = atr.dropna()
        assert (valid_atr >= 0).all()
    
    def test_adx_computation(self, sample_stock_data):
        """Test ADX computation."""
        adx = _compute_adx(sample_stock_data, period=14)
        assert isinstance(adx, pd.Series)
        assert len(adx) == len(sample_stock_data)
        # ADX should be between 0 and 100
        valid_adx = adx.dropna()
        assert (valid_adx >= 0).all()
        assert (valid_adx <= 100).all()
    
    def test_stochastic_computation(self, sample_stock_data):
        """Test Stochastic computation."""
        k, d = _compute_stochastic(sample_stock_data)
        assert isinstance(k, pd.Series)
        assert isinstance(d, pd.Series)
        # K and D should be between 0 and 100
        valid_k = k.dropna()
        valid_d = d.dropna()
        assert (valid_k >= 0).all()
        assert (valid_k <= 100).all()
        assert (valid_d >= 0).all()
        assert (valid_d <= 100).all()
    
    def test_mfi_computation(self, sample_stock_data):
        """Test MFI computation."""
        mfi = _compute_mfi(sample_stock_data, period=14)
        assert isinstance(mfi, pd.Series)
        assert len(mfi) == len(sample_stock_data)
        # MFI should be between 0 and 100
        valid_mfi = mfi.dropna()
        assert (valid_mfi >= 0).all()
        assert (valid_mfi <= 100).all()
    
    def test_williams_r_computation(self, sample_stock_data):
        """Test Williams %R computation."""
        wr = _compute_williams_r(sample_stock_data, period=14)
        assert isinstance(wr, pd.Series)
        assert len(wr) == len(sample_stock_data)
        # Williams %R should be between -100 and 0
        valid_wr = wr.dropna()
        assert (valid_wr >= -100).all()
        assert (valid_wr <= 0).all()
    
    def test_compute_target_days(self, sample_stock_data):
        """Test target computation."""
        result = compute_target_days(sample_stock_data, days=5)
        assert 'target' in result.columns
        # Target should be 0 or 1
        valid_target = result['target'].dropna()
        assert valid_target.isin([0, 1]).all()
    
    def test_filter_correlated_features(self, sample_stock_data):
        """Test correlated feature filtering."""
        result = compute_features(sample_stock_data)
        features = [col for col in result.columns if col not in ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'stock_code', 'target']]
        filtered = filter_correlated_features(result, features, threshold=0.9)
        assert isinstance(filtered, list)
        assert len(filtered) <= len(features)
        # All filtered features should be in original features
        for f in filtered:
            assert f in features


class TestFeatureEngineeringEdgeCases:
    """Test edge cases in feature engineering."""
    
    def test_empty_dataframe(self):
        """Test with empty DataFrame."""
        df = pd.DataFrame(columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
        # compute_features requires at least some data for certain computations
        # So we test that it handles the case gracefully
        try:
            result = compute_features(df)
            assert isinstance(result, pd.DataFrame)
            assert len(result) == 0
        except (ValueError, IndexError):
            # Some functions may raise errors on empty data, which is acceptable
            pass
    
    def test_single_row(self):
        """Test with single row DataFrame."""
        df = pd.DataFrame({
            'Date': ['2024-01-01'],
            'Open': [100],
            'High': [105],
            'Low': [95],
            'Close': [102],
            'Volume': [1000000]
        })
        result = compute_features(df)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1
    
    def test_nan_values_handled(self, sample_stock_data):
        """Test that NaN values are handled properly."""
        # Introduce some NaN values
        df = sample_stock_data.copy()
        df.loc[10, 'Close'] = np.nan
        result = compute_features(df)
        assert isinstance(result, pd.DataFrame)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
