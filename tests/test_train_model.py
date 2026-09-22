"""
Tests for train_model.py - model training functions.
"""
import os
import sys
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def sample_training_data():
    """Create sample training data for testing."""
    np.random.seed(42)
    n = 200
    
    # Generate features
    X = pd.DataFrame({
        'feature_1': np.random.randn(n),
        'feature_2': np.random.randn(n),
        'feature_3': np.random.randn(n),
        'feature_4': np.random.randn(n),
        'feature_5': np.random.randn(n),
    })
    
    # Generate binary target with some pattern
    y = pd.Series(
        (X['feature_1'] + X['feature_2'] * 0.5 + np.random.randn(n) * 0.5 > 0).astype(int),
        name='target'
    )
    
    return X, y


class TestModelTraining:
    """Test model training functions."""
    
    def test_imports(self):
        """Test that train_model module can be imported."""
        import src.train_model as tm
        assert hasattr(tm, 'train_xgboost')
        assert hasattr(tm, 'train_lightgbm')
        assert hasattr(tm, 'train_random_forest')
        assert hasattr(tm, 'train_catboost')
        assert hasattr(tm, 'train_all_models')
        assert hasattr(tm, 'objective_ensemble')
        assert hasattr(tm, '_objective_single')
        assert hasattr(tm, '_apply_smote')
        assert hasattr(tm, '_optimize_thresholds')
        assert hasattr(tm, '_create_blending_ensemble')
    
    def test_has_catboost(self):
        """Test that CatBoost availability is tracked."""
        import src.train_model as tm
        assert hasattr(tm, 'HAS_CATBOOST')
        assert isinstance(tm.HAS_CATBOOST, bool)
    
    def test_train_xgboost(self, sample_training_data):
        """Test XGBoost training."""
        import src.train_model as tm
        X, y = sample_training_data
        model = tm.train_xgboost(X, y)
        assert model is not None
        assert hasattr(model, 'predict')
        assert hasattr(model, 'predict_proba')
        
        # Test prediction
        preds = model.predict(X[:10])
        assert len(preds) == 10
        assert set(preds).issubset({0, 1})
        
        proba = model.predict_proba(X[:10])
        assert proba.shape == (10, 2)
        assert np.abs(proba.sum(axis=1) - 1).max() < 1e-6
    
    def test_train_xgboost_with_eval_set(self, sample_training_data):
        """Test XGBoost accepts eval_set param (ignored, for interface compat)."""
        import src.train_model as tm
        X, y = sample_training_data
        split_idx = int(len(X) * 0.8)
        X_train, X_eval = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_eval = y.iloc[:split_idx], y.iloc[split_idx:]
        
        model = tm.train_xgboost(X_train, y_train, eval_set=(X_eval, y_eval))
        assert model is not None
        preds = model.predict(X_eval[:10])
        assert len(preds) == 10
    
    def test_train_lightgbm(self, sample_training_data):
        """Test LightGBM training."""
        import src.train_model as tm
        X, y = sample_training_data
        model = tm.train_lightgbm(X, y)
        assert model is not None
        assert hasattr(model, 'predict')
        assert hasattr(model, 'predict_proba')
        
        # Test prediction
        preds = model.predict(X[:10])
        assert len(preds) == 10
        assert set(preds).issubset({0, 1})
    
    def test_train_random_forest(self, sample_training_data):
        """Test RandomForest training."""
        import src.train_model as tm
        X, y = sample_training_data
        model = tm.train_random_forest(X, y)
        assert model is not None
        assert hasattr(model, 'predict')
        assert hasattr(model, 'predict_proba')
        
        # Test prediction
        preds = model.predict(X[:10])
        assert len(preds) == 10
        assert set(preds).issubset({0, 1})
    
    def test_train_catboost(self, sample_training_data):
        """Test CatBoost training without eval_set (no early stopping)."""
        import src.train_model as tm
        if not tm.HAS_CATBOOST:
            pytest.skip("CatBoost not installed")
        
        X, y = sample_training_data
        model = tm.train_catboost(X, y)
        assert model is not None
        assert hasattr(model, 'predict')
        assert hasattr(model, 'predict_proba')
        
        # Test prediction
        preds = model.predict(X[:10])
        assert len(preds) == 10
        assert set(preds).issubset({0, 1})
    
    def test_train_catboost_with_eval_set(self, sample_training_data):
        """Test CatBoost training with eval_set for early stopping."""
        import src.train_model as tm
        if not tm.HAS_CATBOOST:
            pytest.skip("CatBoost not installed")
        
        X, y = sample_training_data
        # Split into train/eval (80/20)
        split_idx = int(len(X) * 0.8)
        X_train, X_eval = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_eval = y.iloc[:split_idx], y.iloc[split_idx:]
        
        model = tm.train_catboost(X_train, y_train, eval_set=(X_eval, y_eval))
        assert model is not None
        assert hasattr(model, 'predict')
        assert hasattr(model, 'predict_proba')
        
        # Test prediction
        preds = model.predict(X_eval[:10])
        assert len(preds) == 10
        assert set(preds).issubset({0, 1})
    
    def test_apply_smote_disabled(self, sample_training_data):
        """Test SMOTE when disabled."""
        import src.train_model as tm
        X, y = sample_training_data
        
        with patch.object(tm, 'USE_SMOTE', False):
            X_res, y_res = tm._apply_smote(X, y)
            assert len(X_res) == len(X)
            assert len(y_res) == len(y)
    
    def test_apply_smote_enabled(self, sample_training_data):
        """Test SMOTE when enabled."""
        import src.train_model as tm
        X, y = sample_training_data
        
        with patch.object(tm, 'USE_SMOTE', True):
            X_res, y_res = tm._apply_smote(X, y)
            # SMOTE should increase sample count
            assert len(X_res) >= len(X)
            assert len(y_res) >= len(y)
    
    def test_optimize_thresholds(self, sample_training_data):
        """Test threshold optimization."""
        import src.train_model as tm
        X, y = sample_training_data
        
        # Create some predictions
        np.random.seed(42)
        y_proba = np.random.uniform(0.3, 0.7, len(y))
        
        buy_thresh, sell_thresh, f1 = tm._optimize_thresholds(y, y_proba)
        
        assert 0.5 <= buy_thresh <= 0.7
        assert 0.3 <= sell_thresh <= 0.5
        assert sell_thresh < buy_thresh
        assert 0 <= f1 <= 1
    
    def test_create_blending_ensemble(self, sample_training_data):
        """Test Blending ensemble creation."""
        import src.train_model as tm
        from sklearn.model_selection import TimeSeriesSplit
        import xgboost as xgb
        import lightgbm as lgb
        from sklearn.ensemble import RandomForestClassifier
        
        X, y = sample_training_data
        tscv = TimeSeriesSplit(n_splits=3)
        
        estimators = [
            ('xgb', xgb.XGBClassifier(n_estimators=10, verbosity=0)),
            ('lgb', lgb.LGBMClassifier(n_estimators=10, verbosity=-1)),
            ('rf', RandomForestClassifier(n_estimators=10))
        ]
        
        blending = tm._create_blending_ensemble(estimators, tscv)
        assert blending is not None
        assert hasattr(blending, 'fit')
        assert hasattr(blending, 'predict')
        assert hasattr(blending, 'predict_proba')
        
        # Test fitting and prediction
        blending.fit(X, y)
        preds = blending.predict(X[:10])
        assert len(preds) == 10
        
        proba = blending.predict_proba(X[:10])
        assert proba.shape == (10, 2)


class TestPurgeAndClassWeights:
    """Tests for purge/embargo CV and class-weight toggle."""

    def test_purged_splits_removes_overlapping_training(self):
        """Verify purged splits drop training samples whose label window overlaps validation."""
        import src.train_model as tm
        from sklearn.model_selection import TimeSeriesSplit
        np.random.seed(42)
        n = 750
        X = pd.DataFrame(np.random.randn(n, 5))
        tscv = TimeSeriesSplit(n_splits=5)
        splits = tm._purged_splits(tscv, X, days=20)
        assert len(splits) == 5
        for train_idx, val_idx in splits:
            assert len(train_idx) > 0 and len(val_idx) > 0
            assert set(train_idx).isdisjoint(val_idx)
            val_start = val_idx[0]
            # Every remaining train index must satisfy i + 21 < val_start
            assert (np.array(train_idx) + 21 < val_start).all()

    def test_purged_splits_no_days_keeps_original(self):
        """When days=None, _purged_splits should return original splits unchanged."""
        import src.train_model as tm
        from sklearn.model_selection import TimeSeriesSplit
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(200, 5))
        tscv = TimeSeriesSplit(n_splits=3)
        splits = tm._purged_splits(tscv, X, days=None)
        assert len(splits) == 3
        for train_idx, val_idx in splits:
            # Without purge: train = indices 0..val_start-1
            assert len(train_idx) == val_idx[0]

    def test_purged_splits_small_dataset_fallback(self):
        """When purging leaves <60 train rows, original fold is kept."""
        import src.train_model as tm
        from sklearn.model_selection import TimeSeriesSplit
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(200, 5))
        tscv = TimeSeriesSplit(n_splits=5)
        splits = tm._purged_splits(tscv, X, days=20)
        # Should still return 5 splits (fallback for small folds)
        assert len(splits) == 5

    def test_scale_pos_weight_enabled(self):
        """scale_pos_weight returns minority/majority ratio when USE_CLASS_WEIGHTS=True."""
        import src.train_model as tm
        y = pd.Series([0]*30 + [1]*10)
        with patch.object(tm, 'USE_CLASS_WEIGHTS', True):
            w = tm._scale_pos_weight(y)
        assert abs(w - 3.0) < 0.01

    def test_scale_pos_weight_disabled(self):
        """scale_pos_weight returns 1.0 when USE_CLASS_WEIGHTS=False."""
        import src.train_model as tm
        y = pd.Series([0]*30 + [1]*10)
        with patch.object(tm, 'USE_CLASS_WEIGHTS', False):
            w = tm._scale_pos_weight(y)
        assert w == 1.0

    def test_scale_pos_weight_caps_at_3(self):
        """scale_pos_weight is capped at 3.0 for extreme imbalance."""
        import src.train_model as tm
        y = pd.Series([0]*100 + [1]*5)
        with patch.object(tm, 'USE_CLASS_WEIGHTS', True):
            assert tm._scale_pos_weight(y) == 3.0

    def test_build_default_voting(self, sample_training_data):
        """_build_default_voting returns a VotingClassifier with predict_proba."""
        import src.train_model as tm
        X, y = sample_training_data
        model = tm._build_default_voting(X, y)
        assert hasattr(model, 'predict_proba')
        proba = model.predict_proba(X[:10])
        assert proba.shape == (10, 2)

    def test_walk_forward_backtest(self, sample_training_data, tmp_path):
        """_walk_forward_backtest returns stats dict and saves CSV."""
        import src.train_model as tm
        X, y = sample_training_data
        data = X.copy()
        data['Close'] = 100 * (1 + np.random.randn(len(X)) * 0.01).cumprod()
        data['target'] = y
        with patch.object(tm, 'MODELS_DIR', str(tmp_path)):
            stats = tm._walk_forward_backtest(data, X.columns.tolist(), days=5, timeframe_label='5d')
        assert stats is not None
        assert 'f1' in stats and 'auc' in stats
        assert 0 <= stats['auc'] <= 1
        assert (tmp_path / 'walk_forward_5d.csv').exists()


class TestModelMetadata:
    """Test model metadata and saving."""
    
    def test_roc_curve_saving(self, sample_training_data, tmp_path):
        """Test ROC curve saving."""
        import src.train_model as tm
        
        X, y = sample_training_data
        y_proba = np.random.uniform(0.3, 0.7, len(y))
        
        with patch.object(tm, 'MODELS_DIR', str(tmp_path)):
            tm._save_roc_curve(y, y_proba, 'test', 'xgboost')
            
            # Check that file was created
            roc_path = tmp_path / 'roc_curve_test.png'
            assert roc_path.exists()
    
    def test_cleanup_old_models(self, tmp_path):
        """Test old model cleanup."""
        import src.train_model as tm
        
        # Create some fake model files
        for i in range(10):
            (tmp_path / f'best_model_1d_2024010{i}.pkl').touch()
        
        with patch.object(tm, 'MODELS_DIR', str(tmp_path)):
            tm._cleanup_old_models('1d', keep=5)
            
            # Check that only 5 files remain
            remaining = list(tmp_path.glob('best_model_1d_*.pkl'))
            assert len(remaining) == 5


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
