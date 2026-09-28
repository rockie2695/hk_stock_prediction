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


class TestEnsembleIntrospection:
    """iter_fitted_estimators / has_learned_combiner contract across ensemble shapes.

    Regression: callers unpacked `model.estimators_` as (name, estimator) pairs, but
    sklearn's VotingClassifier/StackingClassifier store a bare list of estimators there.
    The resulting ValueError was swallowed, so dynamic weighting silently did nothing
    and Blending models were excluded from disagreement detection entirely.
    """

    @staticmethod
    def _data(n=300):
        rng = np.random.RandomState(0)
        X = pd.DataFrame({f"f{i}": rng.randn(n) for i in range(4)})
        y = pd.Series((X.f0 + 0.5 * X.f1 + rng.randn(n) * 0.5 > 0).astype(int))
        return X, y

    def test_iter_fitted_estimators_voting(self):
        from sklearn.ensemble import RandomForestClassifier, VotingClassifier
        from src.dynamic_weighting import iter_fitted_estimators

        X, y = self._data()
        model = VotingClassifier(
            estimators=[
                ('rf_a', RandomForestClassifier(n_estimators=10, random_state=42)),
                ('rf_b', RandomForestClassifier(n_estimators=10, random_state=42)),
            ],
            voting='soft',
        ).fit(X, y)

        ests = iter_fitted_estimators(model)
        assert set(ests) == {'rf_a', 'rf_b'}
        # Values must be fitted estimators, not (name, est) pairs
        assert all(hasattr(e, 'predict_proba') for e in ests.values())

    def test_iter_fitted_estimators_stacking(self):
        from sklearn.ensemble import RandomForestClassifier, StackingClassifier
        from sklearn.linear_model import LogisticRegression
        from src.dynamic_weighting import iter_fitted_estimators

        X, y = self._data()
        model = StackingClassifier(
            estimators=[
                ('rf_a', RandomForestClassifier(n_estimators=10, random_state=42)),
                ('rf_b', RandomForestClassifier(n_estimators=10, random_state=42)),
            ],
            final_estimator=LogisticRegression(),
            cv=3,
        ).fit(X, y)

        ests = iter_fitted_estimators(model)
        assert set(ests) == {'rf_a', 'rf_b'}
        assert all(hasattr(e, 'predict_proba') for e in ests.values())

    def test_iter_fitted_estimators_non_ensemble(self):
        from sklearn.ensemble import RandomForestClassifier
        from src.dynamic_weighting import iter_fitted_estimators

        X, y = self._data()
        model = RandomForestClassifier(n_estimators=10, random_state=42).fit(X, y)
        assert iter_fitted_estimators(model) == {}

    def test_has_learned_combiner(self):
        from sklearn.ensemble import RandomForestClassifier, StackingClassifier, VotingClassifier
        from sklearn.linear_model import LogisticRegression
        from src.dynamic_weighting import has_learned_combiner

        X, y = self._data()
        rf = RandomForestClassifier(n_estimators=10, random_state=42)
        voting = VotingClassifier(estimators=[('rf', rf)], voting='soft').fit(X, y)
        stacking = StackingClassifier(
            estimators=[('rf', RandomForestClassifier(n_estimators=10, random_state=42))],
            final_estimator=LogisticRegression(), cv=3,
        ).fit(X, y)

        assert has_learned_combiner(voting) is False
        assert has_learned_combiner(stacking) is True
        assert has_learned_combiner(RandomForestClassifier().fit(X, y)) is False

    def test_dynamic_prediction_preserves_stacking_meta_learner(self):
        """Stacking must return its own meta-learned probability, not a hand-weighted mean."""
        from sklearn.ensemble import RandomForestClassifier, StackingClassifier
        from sklearn.linear_model import LogisticRegression
        from src.dynamic_weighting import compute_dynamic_ensemble_prediction

        X, y = self._data()
        model = StackingClassifier(
            estimators=[
                ('rf_a', RandomForestClassifier(n_estimators=10, random_state=42)),
                ('rf_b', RandomForestClassifier(n_estimators=10, random_state=42)),
            ],
            final_estimator=LogisticRegression(max_iter=500),
            cv=3,
        ).fit(X, y)

        Xq = X.iloc[:20]
        got = compute_dynamic_ensemble_prediction({'model': model}, Xq, 'stock', '5d')
        expected = model.predict_proba(Xq)[:, 1]

        assert got.shape == (20,)
        np.testing.assert_allclose(got, expected)

    def test_dynamic_prediction_blending_uses_meta_model(self):
        """BlendingClassifier must be introspected via named_estimators_, not skipped."""
        import src.train_model as tm
        from sklearn.ensemble import RandomForestClassifier
        from src.dynamic_weighting import (
            compute_dynamic_ensemble_prediction, has_learned_combiner, iter_fitted_estimators
        )
        from sklearn.model_selection import TimeSeriesSplit

        X, y = self._data()
        blend = tm._create_blending_ensemble(
            [
                ('rf_a', RandomForestClassifier(n_estimators=10, random_state=42)),
                ('rf_b', RandomForestClassifier(n_estimators=10, random_state=42)),
            ],
            TimeSeriesSplit(n_splits=3),
            days=1,
        ).fit(X, y)

        ests = iter_fitted_estimators(blend)
        assert set(ests) == {'rf_a', 'rf_b'}
        assert has_learned_combiner(blend) is True

        Xq = X.iloc[:20]
        got = compute_dynamic_ensemble_prediction({'model': blend}, Xq, 'stock', '5d')
        np.testing.assert_allclose(got, blend.predict_proba(Xq)[:, 1])
