"""
Tests for model_monitoring.py - prediction verification and calibration.
"""
import os
import sys
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestVerifyPredictionOutcomes:
    """Tests for _verify_prediction_outcomes helper."""

    def test_buy_correct_when_price_rises(self):
        """Buy prediction is correct when target price > pred price."""
        from src.model_monitoring import _verify_prediction_outcomes
        dates = pd.date_range('2026-01-01', periods=30, freq='B')
        prices = np.linspace(100, 110, 30)  # Rising
        price_df = pd.DataFrame({'Date': dates, 'Close': prices})
        preds = [{'signal': 'Buy', 'confidence': 0.7,
                  'prediction_date': dates[0].strftime('%Y-%m-%d')}]
        outcomes = _verify_prediction_outcomes(preds, price_df, tf_days=5)
        assert len(outcomes) == 1
        assert outcomes[0]['is_correct'] is True

    def test_buy_wrong_when_price_falls(self):
        """Buy prediction is wrong when target price < pred price."""
        from src.model_monitoring import _verify_prediction_outcomes
        dates = pd.date_range('2026-01-01', periods=30, freq='B')
        prices = np.linspace(100, 90, 30)  # Falling
        price_df = pd.DataFrame({'Date': dates, 'Close': prices})
        preds = [{'signal': 'Buy', 'confidence': 0.7,
                  'prediction_date': dates[0].strftime('%Y-%m-%d')}]
        outcomes = _verify_prediction_outcomes(preds, price_df, tf_days=5)
        assert len(outcomes) == 1
        assert outcomes[0]['is_correct'] is False

    def test_sell_correct_when_price_falls(self):
        """Sell prediction is correct when target price < pred price."""
        from src.model_monitoring import _verify_prediction_outcomes
        dates = pd.date_range('2026-01-01', periods=30, freq='B')
        prices = np.linspace(100, 90, 30)  # Falling
        price_df = pd.DataFrame({'Date': dates, 'Close': prices})
        preds = [{'signal': 'Sell', 'confidence': 0.7,
                  'prediction_date': dates[0].strftime('%Y-%m-%d')}]
        outcomes = _verify_prediction_outcomes(preds, price_df, tf_days=5)
        assert len(outcomes) == 1
        assert outcomes[0]['is_correct'] is True

    def test_hold_excluded(self):
        """Hold signals are excluded from outcomes."""
        from src.model_monitoring import _verify_prediction_outcomes
        dates = pd.date_range('2026-01-01', periods=30, freq='B')
        price_df = pd.DataFrame({'Date': dates, 'Close': np.linspace(100, 110, 30)})
        preds = [{'signal': 'Hold', 'confidence': 0.5,
                  'prediction_date': dates[0].strftime('%Y-%m-%d')}]
        outcomes = _verify_prediction_outcomes(preds, price_df, tf_days=5)
        assert len(outcomes) == 0

    def test_empty_price_df_returns_empty(self):
        """Empty price_df returns empty outcomes."""
        from src.model_monitoring import _verify_prediction_outcomes
        preds = [{'signal': 'Buy', 'confidence': 0.7, 'prediction_date': '2026-01-01'}]
        outcomes = _verify_prediction_outcomes(preds, pd.DataFrame(), tf_days=5)
        assert outcomes == []

    def test_out_of_range_date_skipped(self):
        """Prediction date beyond price data is skipped."""
        from src.model_monitoring import _verify_prediction_outcomes
        dates = pd.date_range('2026-01-01', periods=10, freq='B')
        price_df = pd.DataFrame({'Date': dates, 'Close': np.linspace(100, 110, 10)})
        preds = [{'signal': 'Buy', 'confidence': 0.7,
                  'prediction_date': '2026-12-31'}]
        outcomes = _verify_prediction_outcomes(preds, price_df, tf_days=5)
        assert len(outcomes) == 0


class TestCalibration:
    """Tests for ConfidenceCalibrator.calculate_calibration."""

    def test_calibrate_returns_ece(self):
        """calculate_calibration returns calibration_score (ECE) and curve."""
        from src.model_monitoring import ConfidenceCalibrator
        mock_client = MagicMock()

        # Build price data first so we can generate aligned prediction dates
        dates = pd.date_range('2025-06-01', periods=300, freq='B')
        close = 100 + np.random.randn(300).cumsum()
        price_df = pd.DataFrame({'Date': dates, 'Close': close})

        # Generate predictions with dates that fall within the price data range
        preds = []
        for i in range(50):
            # Pick dates from the middle of the price range
            pred_date = dates[100 + i].strftime('%Y-%m-%d')
            preds.append({
                'confidence': 0.5 + (i % 5) * 0.1,
                'signal': 'Buy' if i % 2 == 0 else 'Sell',
                'prediction_date': pred_date,
            })
        query_result = MagicMock()
        query_result.data = preds
        mock_client.table.return_value.select.return_value.eq.return_value.eq.return_value.order.return_value.limit.return_value.execute.return_value = query_result

        calibrator = ConfidenceCalibrator(mock_client)
        with patch('src.model_monitoring.fetch_stock_data', return_value=price_df):
            result = calibrator.calculate_calibration('0700', timeframe='5d', days=300)

        assert 'calibration_score' in result
        assert 'calibration_curve' in result
        if result['calibration_score'] is not None:
            assert 0 <= result['calibration_score'] <= 1

    def test_calibrate_insufficient_data(self):
        """calculate_calibration returns error message when too few predictions."""
        from src.model_monitoring import ConfidenceCalibrator
        mock_client = MagicMock()
        query_result = MagicMock()
        query_result.data = [{'confidence': 0.5, 'signal': 'Buy', 'prediction_date': '2026-01-01'}] * 5
        mock_client.table.return_value.select.return_value.eq.return_value.eq.return_value.order.return_value.limit.return_value.execute.return_value = query_result

        calibrator = ConfidenceCalibrator(mock_client)
        result = calibrator.calculate_calibration('0700', timeframe='5d')
        assert result.get('calibration_score') is None
        assert '不足' in result.get('message', '') or '不足' in result.get('message', '')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
