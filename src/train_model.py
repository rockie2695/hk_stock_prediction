"""
Model training - Optuna hyperparameter tuning + Walk-Forward validation.
Trains both XGBoost and LightGBM, selects the best, saves to models/best_model.pkl.
"""
import os
import sys
import pickle
import pandas as pd
import numpy as np
import optuna
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import f1_score, roc_auc_score
import xgboost as xgb
import lightgbm as lgb

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import STOCK_LIST
from src.data_fetcher import fetch_stock_data
from src.feature_engineering import compute_features, compute_target, FEATURE_COLUMNS
from src.logger import setup_logger

logger = setup_logger('train_model')

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'best_model.pkl')


def prepare_data(stock_codes: list) -> pd.DataFrame:
    """Fetch and combine data for all stock codes."""
    all_data = []
    for code in stock_codes:
        try:
            logger.info(f"Fetching data for {code}...")
            df = fetch_stock_data(code, years=3)
            df = compute_features(df)
            df = compute_target(df)
            df['stock_code'] = code
            all_data.append(df)
            logger.info(f"  {code}: {len(df)} rows")
        except Exception as e:
            logger.error(f"  Failed to fetch {code}: {e}")
            continue

    if not all_data:
        raise RuntimeError("No data fetched for any stock code.")

    combined = pd.concat(all_data, ignore_index=True)
    # Drop rows with NaN in feature columns
    combined = combined.dropna(subset=FEATURE_COLUMNS + ['target'])
    logger.info(f"Combined dataset: {len(combined)} rows, {len(stock_codes)} stocks")
    return combined


def train_xgboost(X_train, y_train, X_val, y_val, trial=None):
    """Train XGBoost with optional Optuna params."""
    if trial:
        params = {
            'n_estimators': trial.suggest_int('xgb_n_estimators', 50, 500),
            'max_depth': trial.suggest_int('xgb_max_depth', 3, 12),
            'learning_rate': trial.suggest_float('xgb_learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('xgb_subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('xgb_colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('xgb_min_child_weight', 1, 10),
            'reg_alpha': trial.suggest_float('xgb_reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('xgb_reg_lambda', 1e-8, 10.0, log=True),
        }
    else:
        params = {}

    model = xgb.XGBClassifier(
        **params,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss',
        verbosity=0
    )
    model.fit(X_train, y_train)
    return model


def train_lightgbm(X_train, y_train, X_val, y_val, trial=None):
    """Train LightGBM with optional Optuna params."""
    if trial:
        params = {
            'n_estimators': trial.suggest_int('lgb_n_estimators', 50, 500),
            'max_depth': trial.suggest_int('lgb_max_depth', 3, 12),
            'learning_rate': trial.suggest_float('lgb_learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('lgb_subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('lgb_colsample_bytree', 0.6, 1.0),
            'min_child_samples': trial.suggest_int('lgb_min_child_samples', 5, 50),
            'reg_alpha': trial.suggest_float('lgb_reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('lgb_reg_lambda', 1e-8, 10.0, log=True),
        }
    else:
        params = {}

    model = lgb.LGBMClassifier(
        **params,
        random_state=42,
        verbosity=-1
    )
    model.fit(X_train, y_train)
    return model


def objective_xgboost(trial, X, y, tscv):
    """Optuna objective for XGBoost."""
    scores = []
    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        model = train_xgboost(X_train, y_train, X_val, y_val, trial=trial)
        preds = model.predict(X_val)
        scores.append(f1_score(y_val, preds, zero_division=0))
    return np.mean(scores)


def objective_lightgbm(trial, X, y, tscv):
    """Optuna objective for LightGBM."""
    scores = []
    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        model = train_lightgbm(X_train, y_train, X_val, y_val, trial=trial)
        preds = model.predict(X_val)
        scores.append(f1_score(y_val, preds, zero_division=0))
    return np.mean(scores)


def train_model(stock_codes: list) -> str:
    """
    Train models for all given stock codes, tune with Optuna,
    and save the best model to models/best_model.pkl.
    """
    os.makedirs(os.path.join(PROJECT_ROOT, 'models'), exist_ok=True)

    # Prepare data
    data = prepare_data(stock_codes)
    X = data[FEATURE_COLUMNS]
    y = data['target']

    logger.info(f"Feature matrix shape: {X.shape}")
    logger.info(f"Target distribution: {y.value_counts().to_dict()}")

    tscv = TimeSeriesSplit(n_splits=5)

    # --- Optuna for XGBoost ---
    logger.info("Running Optuna optimization for XGBoost (50 trials)...")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study_xgb = optuna.create_study(direction='maximize')
    study_xgb.optimize(lambda trial: objective_xgboost(trial, X, y, tscv), n_trials=50)
    logger.info(f"XGBoost best F1 (CV mean): {study_xgb.best_value:.4f}")
    logger.info(f"XGBoost best params: {study_xgb.best_params}")

    # --- Optuna for LightGBM ---
    logger.info("Running Optuna optimization for LightGBM (50 trials)...")
    study_lgb = optuna.create_study(direction='maximize')
    study_lgb.optimize(lambda trial: objective_lightgbm(trial, X, y, tscv), n_trials=50)
    logger.info(f"LightGBM best F1 (CV mean): {study_lgb.best_value:.4f}")
    logger.info(f"LightGBM best params: {study_lgb.best_params}")

    # --- Retrain both on LAST fold and compare ---
    splits = list(tscv.split(X))
    train_idx, val_idx = splits[-1]
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    logger.info("Retraining XGBoost with best params on last fold...")
    xgb_model = train_xgboost(X_train, y_train, X_val, y_val, trial=None)
    # Re-apply best params manually
    best_xgb_params = {k.replace('xgb_', ''): v for k, v in study_xgb.best_params.items()}
    xgb_model = xgb.XGBClassifier(
        **best_xgb_params,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss',
        verbosity=0
    )
    xgb_model.fit(X_train, y_train)
    xgb_preds = xgb_model.predict(X_val)
    xgb_f1 = f1_score(y_val, xgb_preds, zero_division=0)
    xgb_auc = roc_auc_score(y_val, xgb_model.predict_proba(X_val)[:, 1])
    logger.info(f"XGBoost last fold -> F1: {xgb_f1:.4f}, AUC: {xgb_auc:.4f}")

    logger.info("Retraining LightGBM with best params on last fold...")
    best_lgb_params = {k.replace('lgb_', ''): v for k, v in study_lgb.best_params.items()}
    lgb_model = lgb.LGBMClassifier(
        **best_lgb_params,
        random_state=42,
        verbosity=-1
    )
    lgb_model.fit(X_train, y_train)
    lgb_preds = lgb_model.predict(X_val)
    lgb_f1 = f1_score(y_val, lgb_preds, zero_division=0)
    lgb_auc = roc_auc_score(y_val, lgb_model.predict_proba(X_val)[:, 1])
    logger.info(f"LightGBM last fold -> F1: {lgb_f1:.4f}, AUC: {lgb_auc:.4f}")

    # --- Select best model ---
    if xgb_f1 >= lgb_f1:
        best_model = xgb_model
        best_type = 'xgboost'
        best_f1 = xgb_f1
        best_auc = xgb_auc
        logger.info(f"Selected XGBoost (F1={xgb_f1:.4f} >= LightGBM F1={lgb_f1:.4f})")
    else:
        best_model = lgb_model
        best_type = 'lightgbm'
        best_f1 = lgb_f1
        best_auc = lgb_auc
        logger.info(f"Selected LightGBM (F1={lgb_f1:.4f} > XGBoost F1={xgb_f1:.4f})")

    # --- Save model ---
    model_data = {
        'model': best_model,
        'model_type': best_type,
        'feature_columns': FEATURE_COLUMNS,
    }
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model_data, f)

    logger.info(f"Best model saved to {MODEL_PATH}")
    logger.info(f"Model type: {best_type}, F1: {best_f1:.4f}, AUC: {best_auc:.4f}")

    return MODEL_PATH


if __name__ == '__main__':
    logger.info("=== Model Training Started ===")
    logger.info(f"Stock codes: {STOCK_LIST}")
    try:
        path = train_model(STOCK_LIST)
        logger.info(f"Training complete. Model saved at: {path}")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)
