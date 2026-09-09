"""
Tests for predict_upload.py - prediction and upload functions.
"""
import os
import sys
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def sample_prediction_data():
    """Create sample data for prediction testing."""
    np.random.seed(42)
    n = 100
    
    dates = pd.date_range('2024-01-01', periods=n, freq='B')
    base_price = 100
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
    
    # Add some features that predict_upload expects
    df['ret_1d'] = df['Close'].pct_change(1)
    df['ret_5d'] = df['Close'].pct_change(5)
    df['vol_ratio_5d'] = df['Volume'] / df['Volume'].rolling(5).mean()
    df['rsi_14'] = 50 + np.random.randn(n) * 15
    df['macd_diff'] = np.random.randn(n) * 0.5
    df['bb_width'] = np.random.uniform(0.02, 0.08, n)
    df['atr_14'] = np.random.uniform(1, 3, n)
    df['adx'] = np.random.uniform(20, 50, n)
    df['stoch_k'] = np.random.uniform(20, 80, n)
    df['mfi'] = np.random.uniform(30, 70, n)
    df['williams_r'] = np.random.uniform(-80, -20, n)
    
    return df


class TestPredictUpload:
    """Test prediction and upload functions."""
    
    def test_imports(self):
        """Test that predict_upload module can be imported."""
        import src.predict_upload as pu
        assert hasattr(pu, 'load_models')
        assert hasattr(pu, 'predict_stock')
        assert hasattr(pu, 'upload_to_supabase')
        assert hasattr(pu, 'predict_and_upload')
        assert hasattr(pu, 'get_prediction_date')
        assert hasattr(pu, 'get_previous_confidence')
        assert hasattr(pu, 'get_win_rate')
        assert hasattr(pu, 'fetch_market_features')
        assert hasattr(pu, 'get_supabase_client')
    
    def test_get_prediction_date(self):
        """Test prediction date calculation."""
        import src.predict_upload as pu
        
        # Test 1 day ahead
        date_1d = pu.get_prediction_date(1)
        assert isinstance(date_1d, str)
        # Should be a future date
        assert datetime.fromisoformat(date_1d) > datetime.now()
        
        # Test 5 days ahead
        date_5d = pu.get_prediction_date(5)
        assert isinstance(date_5d, str)
        assert datetime.fromisoformat(date_5d) > datetime.now()
        
        # Test 20 days ahead
        date_20d = pu.get_prediction_date(20)
        assert isinstance(date_20d, str)
        assert datetime.fromisoformat(date_20d) > datetime.now()
        
        # 5d should be after 1d
        assert datetime.fromisoformat(date_5d) > datetime.fromisoformat(date_1d)
        
        # 20d should be after 5d
        assert datetime.fromisoformat(date_20d) > datetime.fromisoformat(date_5d)
    
    def test_get_prediction_date_skips_weekends(self):
        """Test that prediction date skips weekends."""
        import src.predict_upload as pu
        
        # Get a date that would be a weekend
        date_str = pu.get_prediction_date(1)
        date = datetime.fromisoformat(date_str)
        
        # Should not be Saturday (5) or Sunday (6)
        assert date.weekday() < 5, f"Prediction date is a weekend: {date_str}"
    
    def test_load_models(self):
        """Test model loading."""
        import src.predict_upload as pu
        
        models = pu.load_models()
        assert isinstance(models, dict)
        
        # Should have models for each timeframe
        for label in ['1d', '5d', '20d']:
            if label in models:
                model_data = models[label]
                assert 'model' in model_data
                assert 'model_type' in model_data
                assert 'feature_columns' in model_data
                assert 'timeframe' in model_data
                assert 'f1_score' in model_data
                assert 'auc_score' in model_data
                assert 'threshold_buy' in model_data
                assert 'threshold_sell' in model_data
    
    def test_supabase_client_singleton(self):
        """Test that Supabase client is singleton."""
        import src.predict_upload as pu
        
        # Reset singleton
        pu._supabase_client = None
        
        with patch.object(pu, 'create_client') as mock_create:
            mock_client = MagicMock()
            mock_create.return_value = mock_client
            
            client1 = pu.get_supabase_client()
            client2 = pu.get_supabase_client()
            
            # Should only create client once
            assert mock_create.call_count == 1
            assert client1 is client2
    
    def test_predict_stock_with_mock_models(self, sample_prediction_data):
        """Test stock prediction with mock models."""
        import src.predict_upload as pu
        from unittest.mock import MagicMock
        
        # Create mock models
        mock_models = {}
        for label in ['1d', '5d', '20d']:
            mock_model = MagicMock()
            mock_model.predict_proba.return_value = np.array([[0.3, 0.7]])
            
            mock_models[label] = {
                'model': mock_model,
                'model_type': 'xgboost',
                'feature_columns': sample_prediction_data.columns.tolist(),
                'timeframe': label,
                'days': pu.TIMEFRAMES[label],
                'f1_score': 0.6,
                'auc_score': 0.55,
                'threshold_buy': 0.55,
                'threshold_sell': 0.45
            }
        
        with patch.object(pu, 'fetch_stock_data', return_value=sample_prediction_data):
            with patch.object(pu, 'fetch_market_features', return_value=pd.DataFrame()):
                with patch.object(pu, 'get_supabase_client') as mock_get_client:
                    mock_client = MagicMock()
                    mock_client.table.return_value.select.return_value.order.return_value.limit.return_value.execute.return_value.data = []
                    mock_get_client.return_value = mock_client
                    
                    results = pu.predict_stock('0700', mock_models)
                    
                    assert isinstance(results, list)
                    assert len(results) == 3  # One for each timeframe
                    
                    for result in results:
                        assert 'stock_code' in result
                        assert 'prediction_date' in result
                        assert 'timeframe' in result
                        assert 'signal' in result
                        assert 'confidence' in result
                        assert 'model_type' in result
                        assert 'f1_score' in result
                        assert 'auc_score' in result
                        assert 'expected_return' in result
                        assert 'risk_reward' in result
                        assert 'stop_loss' in result
                        assert 'take_profit' in result
                        assert 'confidence_trend' in result
                        assert 'win_rate' in result
                        assert 'threshold_buy' in result
                        assert 'threshold_sell' in result
    
    def test_signal_determination(self, sample_prediction_data):
        """Test that signals are determined correctly based on thresholds."""
        import src.predict_upload as pu
        from unittest.mock import MagicMock
        
        # Create mock model with known probability
        mock_model = MagicMock()
        mock_model.predict_proba.return_value = np.array([[0.4, 0.6]])  # 60% buy prob
        
        mock_models = {
            '1d': {
                'model': mock_model,
                'model_type': 'xgboost',
                'feature_columns': sample_prediction_data.columns.tolist(),
                'timeframe': '1d',
                'days': 1,
                'f1_score': 0.6,
                'auc_score': 0.55,
                'threshold_buy': 0.55,
                'threshold_sell': 0.45
            }
        }
        
        with patch.object(pu, 'fetch_stock_data', return_value=sample_prediction_data):
            with patch.object(pu, 'fetch_market_features', return_value=pd.DataFrame()):
                with patch.object(pu, 'get_supabase_client') as mock_get_client:
                    mock_client = MagicMock()
                    mock_client.table.return_value.select.return_value.order.return_value.limit.return_value.execute.return_value.data = []
                    mock_get_client.return_value = mock_client
                    
                    results = pu.predict_stock('0700', mock_models)
                    
                    # With 60% buy prob and threshold 0.55, signal should be Buy
                    assert results[0]['signal'] == 'Buy'
                    assert results[0]['confidence'] == 0.6
    
    def test_upload_to_supabase(self):
        """Test Supabase upload."""
        import src.predict_upload as pu
        
        records = [
            {
                'stock_code': '0700',
                'prediction_date': '2024-01-10',
                'timeframe': '1d',
                'signal': 'Buy',
                'confidence': 0.65,
                'model_version': '2024-01-01',
                'model_type': 'xgboost',
                'f1_score': 0.6,
                'auc_score': 0.55,
                'expected_return': 2.5,
                'risk_reward': 1.2,
                'stop_loss': -3.0,
                'take_profit': 3.75,
                'confidence_trend': '↑',
                'win_rate': 65.0,
                'threshold_buy': 0.55,
                'threshold_sell': 0.45
            }
        ]
        
        with patch.object(pu, 'get_supabase_client') as mock_get_client:
            mock_client = MagicMock()
            mock_get_client.return_value = mock_client
            
            success, fail = pu.upload_to_supabase(records)
            
            assert success == 1
            assert fail == 0
            mock_client.table.assert_called_with('stock_predictions')
    
    def test_upload_empty_records(self):
        """Test upload with empty records."""
        import src.predict_upload as pu
        
        success, fail = pu.upload_to_supabase([])
        assert success == 0
        assert fail == 0


class TestPredictUploadEdgeCases:
    """Test edge cases in prediction."""
    
    def test_fetch_market_features(self):
        """Test market features fetching."""
        import src.predict_upload as pu
        
        # This will actually try to fetch data, so we mock yfinance
        with patch('yfinance.download') as mock_download:
            # Create mock HSI data
            mock_hsi = pd.DataFrame({
                'Close': np.random.randn(90).cumsum() + 20000
            }, index=pd.date_range('2024-01-01', periods=90))
            
            # Create mock FX data
            mock_fx = pd.DataFrame({
                'Close': np.random.randn(90).cumsum() + 7.8
            }, index=pd.date_range('2024-01-01', periods=90))
            
            mock_download.side_effect = [mock_hsi, mock_fx]
            
            result = pu.fetch_market_features()
            
            assert isinstance(result, pd.DataFrame)
            # Should have hsi_ret_5d and hsi_ret_20d columns if data is sufficient
            if len(result) >= 20:
                assert 'hsi_ret_5d' in result.columns
                assert 'hsi_ret_20d' in result.columns


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
