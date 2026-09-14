"""Tests for online learner module."""
import os
import sys
import pandas as pd
import numpy as np
import pytest
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestOnlineLearner:
    """Tests for incremental model updates."""
    
    def test_import(self):
        from src.online_learner import should_use_online_learning, online_update
        assert callable(should_use_online_learning)
        assert callable(online_update)
    
    def test_should_use_online_learning_no_model(self):
        from src.online_learner import should_use_online_learning
        
        # Should return False when no model exists
        assert should_use_online_learning('nonexistent_1d') == False
    
    def test_should_use_online_learning_old_model(self):
        from src.online_learner import should_use_online_learning
        
        import tempfile
        import pickle
        
        # Create a temporary old model file
        model_data = {
            'model': None,
            'model_type': 'test',
            'feature_columns': ['f1', 'f2'],
        }
        
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            pickle.dump(model_data, f)
            temp_path = f.name
        
        try:
            # Modify the file timestamp to be 10 days old
            old_time = datetime.now().timestamp() - (10 * 24 * 3600)
            os.utime(temp_path, (old_time, old_time))
            
            # Mock the path
            import src.online_learner as ol
            original_dir = ol.MODELS_DIR
            ol.MODELS_DIR = os.path.dirname(temp_path)
            
            # Rename to match expected pattern
            new_path = os.path.join(ol.MODELS_DIR, 'best_model_1d.pkl')
            os.rename(temp_path, new_path)
            
            result = should_use_online_learning('1d')
            assert result == True
            
            os.rename(new_path, temp_path)
            ol.MODELS_DIR = original_dir
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_online_update_no_model(self):
        from src.online_learner import online_update
        
        result = online_update(['0700'], 'nonexistent_1d', 1)
        assert result == False
