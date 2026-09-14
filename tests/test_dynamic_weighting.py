"""Tests for dynamic weighting module."""
import os
import sys
import pandas as pd
import numpy as np
import pytest
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestDynamicWeighting:
    """Tests for dynamic ensemble weighting."""
    
    def test_import(self):
        from src.dynamic_weighting import (
            load_dynamic_weights, save_dynamic_weights,
            update_model_weights, get_dynamic_weights,
            evaluate_model_performance
        )
        assert callable(load_dynamic_weights)
        assert callable(save_dynamic_weights)
        assert callable(update_model_weights)
        assert callable(get_dynamic_weights)
        assert callable(evaluate_model_performance)
    
    def test_default_weights(self):
        from src.dynamic_weighting import get_dynamic_weights
        
        weights = get_dynamic_weights('nonexistent_stock', '5d')
        
        assert 'xgb' in weights
        assert 'lgb' in weights
        assert 'rf' in weights
        assert 'cb' in weights
        assert abs(sum(weights.values()) - 1.0) < 0.01
    
    def test_update_weights(self):
        from src.dynamic_weighting import update_model_weights, get_dynamic_weights
        
        accuracies = {'xgb': 0.7, 'lgb': 0.65, 'rf': 0.6, 'cb': 0.72}
        
        weights = update_model_weights('test_stock', '5d', accuracies, alpha=0.5)
        
        assert abs(sum(weights.values()) - 1.0) < 0.01
        assert weights['cb'] > weights['rf']  # cb had higher accuracy
    
    def test_evaluate_performance(self):
        from src.dynamic_weighting import evaluate_model_performance
        
        y_true = np.array([0, 1, 1, 0, 1, 1, 0, 0, 1, 1])
        y_pred = np.array([0, 1, 0, 0, 1, 1, 0, 1, 1, 1])
        
        metrics = evaluate_model_performance(y_true, y_pred)
        
        assert 'f1' in metrics
        assert 'accuracy' in metrics
        assert 0 <= metrics['f1'] <= 1
        assert 0 <= metrics['accuracy'] <= 1
    
    def test_weights_persistence(self):
        from src.dynamic_weighting import update_model_weights, save_dynamic_weights, load_dynamic_weights
        import tempfile
        
        accuracies = {'xgb': 0.7, 'lgb': 0.65, 'rf': 0.6, 'cb': 0.72}
        update_model_weights('persist_stock', '5d', accuracies, alpha=0.5)
        
        weights = load_dynamic_weights()
        assert 'persist_stock_5d' in weights
