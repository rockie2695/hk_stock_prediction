"""
Model training - Optuna hyperparameter tuning + Walk-Forward validation.
Trains XGBoost and LightGBM for 3 timeframes (1d, 5d, 20d).
Includes market context features (HSI index, USD/HKD).
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
from src.feature_engineering import compute_features, compute_target_days, compute_target_threshold, FEATURE_COLUMNS
from src.logger import setup_logger

logger = setup_logger('train_model')

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')

# Timeframes: label -> days ahead
TIMEFRAMES = {'1d': 1, '5d': 5, '20d': 20}


def fetch_market_data(years: int = 3) -> pd.DataFrame:
    """Fetch HSI index and USD/HKD as market context."""
    import yfinance as yf
    from datetime import datetime, timedelta
    import pytz

    end_date = datetime.now(pytz.timezone('Asia/Hong_Kong'))
    start_date = end_date - timedelta(days=years * 365)

    logger.info("Fetching market context (HSI index, USD/HKD)...")

    data = {}

    # HSI index (^HSI)
    try:
        hsi = yf.download('^HSI', start=start_date, end=end_date, progress=False)
        if not hsi.empty:
            # Flatten MultiIndex columns if present
            if isinstance(hsi.columns, pd.MultiIndex):
                hsi.columns = hsi.columns.get_level_values(0)
            data['hsi_close'] = hsi['Close']
            logger.info(f"  HSI: {len(hsi)} rows")
    except Exception as e:
        logger.warning(f"  HSI fetch failed: {e}")

    # USD/HKD
    try:
        fx = yf.download('HKD=X', start=start_date, end=end_date, progress=False)
        if not fx.empty:
            if isinstance(fx.columns, pd.MultiIndex):
                fx.columns = fx.columns.get_level_values(0)
            data['usdhkd'] = fx['Close']
            logger.info(f"  USD/HKD: {len(fx)} rows")
    except Exception as e:
        logger.warning(f"  USD/HKD fetch failed: {e}")

    if not data:
        logger.warning("No market data fetched")
        return pd.DataFrame()

    market_df = pd.DataFrame(data)
    market_df = market_df.ffill()  # forward fill missing dates

    # Normalize timezone to match stock data
    if market_df.index.tz is not None:
        market_df.index = market_df.index.tz_localize(None).normalize()

    # Compute market features
    if 'hsi_close' in market_df.columns:
        market_df['hsi_ret_5d'] = market_df['hsi_close'].pct_change(5)
        market_df['hsi_ret_20d'] = market_df['hsi_close'].pct_change(20)
    if 'usdhkd' in market_df.columns:
        market_df['usdhkd_change'] = market_df['usdhkd'].pct_change(5)

    # Drop raw columns
    market_df = market_df.drop(columns=['hsi_close', 'usdhkd'], errors='ignore')

    return market_df


def prepare_data(stock_codes: list, days: int) -> pd.DataFrame:
    """Fetch and combine data for all stock codes with N-day target."""
    # Fetch market context
    market_df = fetch_market_data()

    all_data = []
    for code in stock_codes:
        try:
            logger.info(f"  Fetching data for {code}...")
            df = fetch_stock_data(code, years=3)
            df = compute_features(df)

            # Use normal target (no threshold)
            df = compute_target_days(df, days)

            df['stock_code'] = code

            # Ensure Date is datetime and normalized
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None).dt.normalize()

            # Join market context
            if not market_df.empty:
                market_with_date = market_df.reset_index()
                if 'Date' not in market_with_date.columns:
                    market_with_date = market_with_date.rename(columns={market_with_date.columns[0]: 'Date'})
                market_with_date['Date'] = pd.to_datetime(market_with_date['Date']).dt.tz_localize(None).dt.normalize()
                df = df.merge(market_with_date, on='Date', how='left')
                df = df.ffill()

            all_data.append(df)
            logger.info(f"    {code}: {len(df)} rows")
        except Exception as e:
            logger.error(f"    Failed to fetch {code}: {e}")
            continue

    if not all_data:
        raise RuntimeError("No data fetched for any stock code.")

    combined = pd.concat(all_data, ignore_index=True)

    # Build feature list (base + available market features)
    available_features = FEATURE_COLUMNS.copy()
    for col in ['hsi_ret_5d', 'hsi_ret_20d', 'usdhkd_change']:
        if col in combined.columns:
            available_features.append(col)

    # Drop rows with NaN in features or target
    combined = combined.dropna(subset=available_features + ['target'])

    return combined, available_features


def train_xgboost(X_train, y_train, trial=None):
    """Train XGBoost with optional Optuna params."""
    # Compute class weights (capped to avoid over-correction)
    n0 = (y_train == 0).sum()
    n1 = (y_train == 1).sum()
    scale_pos_weight = min(n0 / n1, 3.0) if n1 > 0 else 1  # cap at 3x

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
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss',
        verbosity=0
    )
    model.fit(X_train, y_train)
    return model


def train_lightgbm(X_train, y_train, trial=None):
    """Train LightGBM with optional Optuna params."""
    # Compute class weights (capped to avoid over-correction)
    n0 = (y_train == 0).sum()
    n1 = (y_train == 1).sum()
    scale_pos_weight = min(n0 / n1, 3.0) if n1 > 0 else 1  # cap at 3x

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
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        verbosity=-1
    )
    model.fit(X_train, y_train)
    return model


def objective_xgboost(trial, X, y, tscv):
    scores = []
    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        model = train_xgboost(X_train, y_train, trial=trial)
        preds = model.predict(X_val)
        scores.append(f1_score(y_val, preds, zero_division=0))
    return np.mean(scores)


def objective_lightgbm(trial, X, y, tscv):
    scores = []
    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        model = train_lightgbm(X_train, y_train, trial=trial)
        preds = model.predict(X_val)
        scores.append(f1_score(y_val, preds, zero_division=0))
    return np.mean(scores)


def train_single_timeframe(stock_codes: list, timeframe_label: str, days: int):
    """Train and save model for one timeframe."""
    logger.info(f"\n{'='*50}")
    logger.info(f"Training model for {timeframe_label} ({days}-day ahead)")
    logger.info(f"{'='*50}")

    # Prepare data with normal target
    data, available_features = prepare_data(stock_codes, days)

    # Use all available features
    X = data[available_features]
    y = data['target']

    logger.info(f"Dataset: {len(X)} rows, {len(available_features)} features")
    logger.info(f"Target distribution: {y.value_counts().to_dict()}")

    tscv = TimeSeriesSplit(n_splits=5)

    # Optuna for XGBoost
    logger.info(f"Optuna XGBoost (50 trials)...")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study_xgb = optuna.create_study(direction='maximize')
    study_xgb.optimize(lambda trial: objective_xgboost(trial, X, y, tscv), n_trials=50)
    logger.info(f"  XGBoost best F1 (CV): {study_xgb.best_value:.4f}")

    # Optuna for LightGBM
    logger.info(f"Optuna LightGBM (50 trials)...")
    study_lgb = optuna.create_study(direction='maximize')
    study_lgb.optimize(lambda trial: objective_lightgbm(trial, X, y, tscv), n_trials=50)
    logger.info(f"  LightGBM best F1 (CV): {study_lgb.best_value:.4f}")

    # Retrain on LAST fold
    splits = list(tscv.split(X))
    train_idx, val_idx = splits[-1]
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    # XGBoost
    best_xgb_params = {k.replace('xgb_', ''): v for k, v in study_xgb.best_params.items()}
    xgb_model = xgb.XGBClassifier(**best_xgb_params, random_state=42, use_label_encoder=False, eval_metric='logloss', verbosity=0)
    xgb_model.fit(X_train, y_train)
    xgb_f1 = f1_score(y_val, xgb_model.predict(X_val), zero_division=0)
    xgb_auc = roc_auc_score(y_val, xgb_model.predict_proba(X_val)[:, 1])

    # LightGBM
    best_lgb_params = {k.replace('lgb_', ''): v for k, v in study_lgb.best_params.items()}
    lgb_model = lgb.LGBMClassifier(**best_lgb_params, random_state=42, verbosity=-1)
    lgb_model.fit(X_train, y_train)
    lgb_f1 = f1_score(y_val, lgb_model.predict(X_val), zero_division=0)
    lgb_auc = roc_auc_score(y_val, lgb_model.predict_proba(X_val)[:, 1])

    # Select best
    if xgb_f1 >= lgb_f1:
        best_model = xgb_model
        best_type = 'xgboost'
        best_f1, best_auc = xgb_f1, xgb_auc
    else:
        best_model = lgb_model
        best_type = 'lightgbm'
        best_f1, best_auc = lgb_f1, lgb_auc

    logger.info(f"Winner: {best_type} (F1={best_f1:.4f}, AUC={best_auc:.4f})")

    # Feature importance
    importance = pd.DataFrame({
        'feature': available_features,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    logger.info(f"\nTop 10 features:")
    for _, row in importance.head(10).iterrows():
        logger.info(f"  {row['feature']}: {row['importance']:.4f}")

    # Save
    model_path = os.path.join(MODELS_DIR, f'best_model_{timeframe_label}.pkl')
    model_data = {
        'model': best_model,
        'model_type': best_type,
        'feature_columns': available_features,
        'timeframe': timeframe_label,
        'days': days,
        'f1_score': best_f1,
        'auc_score': best_auc,
    }
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
    logger.info(f"Saved: {model_path}")

    return model_path, best_type, best_f1, best_auc


def train_all_models(stock_codes: list):
    """Train models for all timeframes."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    logger.info("=== Multi-Timeframe Model Training ===")

    results = {}
    for label, days in TIMEFRAMES.items():
        path, model_type, f1, auc = train_single_timeframe(stock_codes, label, days)
        results[label] = {'path': path, 'type': model_type, 'f1': f1, 'auc': auc}

    # Summary
    logger.info("\n" + "="*50)
    logger.info("TRAINING SUMMARY")
    logger.info("="*50)
    for label, r in results.items():
        logger.info(f"  {label}: {r['type']} | F1={r['f1']:.4f} | AUC={r['auc']:.4f}")
    logger.info("="*50)


if __name__ == '__main__':
    logger.info("=== Model Training Started ===")
    logger.info(f"Stock codes: {STOCK_LIST}")
    try:
        train_all_models(STOCK_LIST)
        logger.info("All models trained successfully.")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)
