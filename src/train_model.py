"""
Model training - Optuna hyperparameter tuning + Walk-Forward validation.
Supports Voting/Stacking ensemble of XGBoost, LightGBM, RandomForest.
Includes SMOTE for class imbalance and comprehensive metrics (F1, AUC, Precision, Recall).
"""
import os
import sys
import pickle
import pandas as pd
import numpy as np
import optuna
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score, roc_curve
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import STOCK_LIST, USE_ENSEMBLE, USE_STACKING, USE_SMOTE
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
    market_df = market_df.ffill()

    if market_df.index.tz is not None:
        market_df.index = market_df.index.tz_localize(None).normalize()

    if 'hsi_close' in market_df.columns:
        market_df['hsi_ret_5d'] = market_df['hsi_close'].pct_change(5)
        market_df['hsi_ret_20d'] = market_df['hsi_close'].pct_change(20)
    if 'usdhkd' in market_df.columns:
        market_df['usdhkd_change'] = market_df['usdhkd'].pct_change(5)

    market_df = market_df.drop(columns=['hsi_close', 'usdhkd'], errors='ignore')

    return market_df


def prepare_data(stock_codes: list, days: int) -> pd.DataFrame:
    """Fetch and combine data for all stock codes with N-day target."""
    market_df = fetch_market_data()

    all_data = []
    for code in stock_codes:
        try:
            logger.info(f"  Fetching data for {code}...")
            df = fetch_stock_data(code, years=3)
            df = compute_features(df)
            df = compute_target_days(df, days)
            df['stock_code'] = code

            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None).dt.normalize()

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

    available_features = FEATURE_COLUMNS.copy()
    for col in ['hsi_ret_5d', 'hsi_ret_20d', 'usdhkd_change']:
        if col in combined.columns:
            available_features.append(col)

    combined = combined.dropna(subset=available_features + ['target'])

    return combined, available_features


def _apply_smote(X_train, y_train):
    """Apply SMOTE to training data only."""
    if not USE_SMOTE:
        return X_train, y_train
    try:
        from imblearn.over_sampling import SMOTE
        smote = SMOTE(random_state=42, sampling_strategy='auto')
        X_res, y_res = smote.fit_resample(X_train, y_train)
        logger.info(f"    SMOTE applied: {len(X_train)} -> {len(X_res)} samples")
        return X_res, y_res
    except ImportError:
        logger.warning("    imblearn not installed, skipping SMOTE")
        return X_train, y_train
    except Exception as e:
        logger.warning(f"    SMOTE failed: {e}")
        return X_train, y_train


def train_xgboost(X_train, y_train, trial=None):
    """Train XGBoost with optional Optuna params."""
    n0 = (y_train == 0).sum()
    n1 = (y_train == 1).sum()
    scale_pos_weight = min(n0 / n1, 3.0) if n1 > 0 else 1

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
    n0 = (y_train == 0).sum()
    n1 = (y_train == 1).sum()
    scale_pos_weight = min(n0 / n1, 3.0) if n1 > 0 else 1

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


def train_random_forest(X_train, y_train, trial=None):
    """Train RandomForest with optional Optuna params."""
    if trial:
        params = {
            'n_estimators': trial.suggest_int('rf_n_estimators', 50, 500),
            'max_depth': trial.suggest_int('rf_max_depth', 3, 20),
            'min_samples_split': trial.suggest_int('rf_min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('rf_min_samples_leaf', 1, 10),
            'max_features': trial.suggest_float('rf_max_features', 0.3, 1.0),
        }
    else:
        params = {}

    model = RandomForestClassifier(
        **params,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model


def _save_roc_curve(y_true, y_proba, timeframe_label, model_name='ensemble'):
    """Save ROC curve plot."""
    try:
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        auc = roc_auc_score(y_true, y_proba)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, 'b-', label=f'{model_name} (AUC = {auc:.4f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {timeframe_label}')
        plt.legend()
        plt.grid(True)
        roc_path = os.path.join(MODELS_DIR, f'roc_curve_{timeframe_label}.png')
        plt.savefig(roc_path, dpi=100)
        plt.close()
        logger.info(f"  ROC curve saved: {roc_path}")
    except Exception as e:
        logger.warning(f"  Failed to save ROC curve: {e}")


def objective_ensemble(trial, X, y, tscv):
    """Optuna objective for ensemble: tune individual model params + voting weights."""
    # Tune XGBoost
    xgb_params = {
        'n_estimators': trial.suggest_int('xgb_n_estimators', 50, 500),
        'max_depth': trial.suggest_int('xgb_max_depth', 3, 12),
        'learning_rate': trial.suggest_float('xgb_learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('xgb_subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('xgb_colsample_bytree', 0.6, 1.0),
    }
    # Tune LightGBM
    lgb_params = {
        'n_estimators': trial.suggest_int('lgb_n_estimators', 50, 500),
        'max_depth': trial.suggest_int('lgb_max_depth', 3, 12),
        'learning_rate': trial.suggest_float('lgb_learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('lgb_subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('lgb_colsample_bytree', 0.6, 1.0),
    }
    # Tune RandomForest
    rf_params = {
        'n_estimators': trial.suggest_int('rf_n_estimators', 50, 500),
        'max_depth': trial.suggest_int('rf_max_depth', 3, 20),
        'min_samples_split': trial.suggest_int('rf_min_samples_split', 2, 20),
    }
    # Tune voting weights
    w1 = trial.suggest_float('w_xgb', 0.1, 2.0)
    w2 = trial.suggest_float('w_lgb', 0.1, 2.0)
    w3 = trial.suggest_float('w_rf', 0.1, 2.0)

    n0 = (y == 0).sum()
    n1 = (y == 1).sum()
    scale_pos_weight = min(n0 / n1, 3.0) if n1 > 0 else 1

    xgb_model = xgb.XGBClassifier(**xgb_params, scale_pos_weight=scale_pos_weight, random_state=42, use_label_encoder=False, eval_metric='logloss', verbosity=0)
    lgb_model = lgb.LGBMClassifier(**lgb_params, scale_pos_weight=scale_pos_weight, random_state=42, verbosity=-1)
    rf_model = RandomForestClassifier(**rf_params, random_state=42, n_jobs=-1)

    if USE_STACKING:
        estimator_list = [('xgb', xgb_model), ('lgb', lgb_model), ('rf', rf_model)]
        ensemble = StackingClassifier(
            estimators=estimator_list,
            final_estimator=LogisticRegression(random_state=42),
            cv=3,
            passthrough=False
        )
    else:
        ensemble = VotingClassifier(
            estimators=[('xgb', xgb_model), ('lgb', lgb_model), ('rf', rf_model)],
            voting='soft',
            weights=[w1, w2, w3]
        )

    scores = []
    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        X_train_sm, y_train_sm = _apply_smote(X_train, y_train)

        ensemble.fit(X_train_sm, y_train_sm)
        preds = ensemble.predict(X_val)
        scores.append(f1_score(y_val, preds, zero_division=0))
    return np.mean(scores)


def train_single_timeframe(stock_codes: list, timeframe_label: str, days: int):
    """Train and save model for one timeframe."""
    logger.info(f"\n{'='*50}")
    logger.info(f"Training model for {timeframe_label} ({days}-day ahead)")
    logger.info(f"Ensemble: {USE_ENSEMBLE}, Stacking: {USE_STACKING}, SMOTE: {USE_SMOTE}")
    logger.info(f"{'='*50}")

    data, available_features = prepare_data(stock_codes, days)

    X = data[available_features]
    y = data['target']

    logger.info(f"Dataset: {len(X)} rows, {len(available_features)} features")
    logger.info(f"Target distribution: {y.value_counts().to_dict()}")

    tscv = TimeSeriesSplit(n_splits=5)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    if USE_ENSEMBLE:
        # Train ensemble
        logger.info("Optuna Ensemble (50 trials)...")
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: objective_ensemble(trial, X, y, tscv), n_trials=50)
        logger.info(f"  Ensemble best F1 (CV): {study.best_value:.4f}")

        # Retrain on LAST fold
        splits = list(tscv.split(X))
        train_idx, val_idx = splits[-1]
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        X_train_sm, y_train_sm = _apply_smote(X_train, y_train)

        best_p = study.best_params
        n0 = (y_train_sm == 0).sum()
        n1 = (y_train_sm == 1).sum()
        scale_pos_weight = min(n0 / n1, 3.0) if n1 > 0 else 1

        xgb_model = xgb.XGBClassifier(
            n_estimators=best_p['xgb_n_estimators'], max_depth=best_p['xgb_max_depth'],
            learning_rate=best_p['xgb_learning_rate'], subsample=best_p['xgb_subsample'],
            colsample_bytree=best_p['xgb_colsample_bytree'],
            scale_pos_weight=scale_pos_weight, random_state=42, use_label_encoder=False, eval_metric='logloss', verbosity=0)
        lgb_model = lgb.LGBMClassifier(
            n_estimators=best_p['lgb_n_estimators'], max_depth=best_p['lgb_max_depth'],
            learning_rate=best_p['lgb_learning_rate'], subsample=best_p['lgb_subsample'],
            colsample_bytree=best_p['lgb_colsample_bytree'],
            scale_pos_weight=scale_pos_weight, random_state=42, verbosity=-1)
        rf_model = RandomForestClassifier(
            n_estimators=best_p['rf_n_estimators'], max_depth=best_p['rf_max_depth'],
            min_samples_split=best_p['rf_min_samples_split'], random_state=42, n_jobs=-1)

        if USE_STACKING:
            ensemble = StackingClassifier(
                estimators=[('xgb', xgb_model), ('lgb', lgb_model), ('rf', rf_model)],
                final_estimator=LogisticRegression(random_state=42), cv=3, passthrough=False)
            model_type = 'stacking'
        else:
            w1, w2, w3 = best_p['w_xgb'], best_p['w_lgb'], best_p['w_rf']
            ensemble = VotingClassifier(
                estimators=[('xgb', xgb_model), ('lgb', lgb_model), ('rf', rf_model)],
                voting='soft', weights=[w1, w2, w3])
            model_type = 'voting'

        ensemble.fit(X_train_sm, y_train_sm)
        preds = ensemble.predict(X_val)
        proba = ensemble.predict_proba(X_val)[:, 1]
        best_f1 = f1_score(y_val, preds, zero_division=0)
        best_auc = roc_auc_score(y_val, proba)
        precision = precision_score(y_val, preds, zero_division=0)
        recall = recall_score(y_val, preds, zero_division=0)

        best_model = ensemble
        _save_roc_curve(y_val, proba, timeframe_label, model_type)

    else:
        # Single model selection
        logger.info("Training individual models...")

        # Optuna for XGBoost
        logger.info("Optuna XGBoost (50 trials)...")
        study_xgb = optuna.create_study(direction='maximize')
        study_xgb.optimize(lambda trial: _objective_single(train_xgboost, trial, X, y, tscv), n_trials=50)
        logger.info(f"  XGBoost best F1: {study_xgb.best_value:.4f}")

        # Optuna for LightGBM
        logger.info("Optuna LightGBM (50 trials)...")
        study_lgb = optuna.create_study(direction='maximize')
        study_lgb.optimize(lambda trial: _objective_single(train_lightgbm, trial, X, y, tscv), n_trials=50)
        logger.info(f"  LightGBM best F1: {study_lgb.best_value:.4f}")

        splits = list(tscv.split(X))
        train_idx, val_idx = splits[-1]
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        X_train_sm, y_train_sm = _apply_smote(X_train, y_train)

        # XGBoost
        best_xgb_params = {k.replace('xgb_', ''): v for k, v in study_xgb.best_params.items()}
        xgb_model = train_xgboost(X_train_sm, y_train_sm, trial=None)
        xgb_model.set_params(**best_xgb_params)
        xgb_model.fit(X_train_sm, y_train_sm)
        xgb_f1 = f1_score(y_val, xgb_model.predict(X_val), zero_division=0)
        xgb_auc = roc_auc_score(y_val, xgb_model.predict_proba(X_val)[:, 1])

        # LightGBM
        best_lgb_params = {k.replace('lgb_', ''): v for k, v in study_lgb.best_params.items()}
        lgb_model = train_lightgbm(X_train_sm, y_train_sm, trial=None)
        lgb_model.set_params(**best_lgb_params)
        lgb_model.fit(X_train_sm, y_train_sm)
        lgb_f1 = f1_score(y_val, lgb_model.predict(X_val), zero_division=0)
        lgb_auc = roc_auc_score(y_val, lgb_model.predict_proba(X_val)[:, 1])

        if xgb_f1 >= lgb_f1:
            best_model = xgb_model
            best_type = 'xgboost'
            best_f1, best_auc = xgb_f1, xgb_auc
        else:
            best_model = lgb_model
            best_type = 'lightgbm'
            best_f1, best_auc = lgb_f1, lgb_auc

        model_type = best_type
        precision = precision_score(y_val, best_model.predict(X_val), zero_division=0)
        recall = recall_score(y_val, best_model.predict(X_val), zero_division=0)

        _save_roc_curve(y_val, best_model.predict_proba(X_val)[:, 1], timeframe_label, model_type)

    logger.info(f"Winner: {model_type}")
    logger.info(f"  F1={best_f1:.4f}, AUC={best_auc:.4f}, Precision={precision:.4f}, Recall={recall:.4f}")

    # Feature importance (for ensemble, use xgb feature importances)
    if hasattr(best_model, 'feature_importances_'):
        importances = best_model.feature_importances_
    elif hasattr(best_model, 'estimators_'):
        # Voting/Stacking: average importances from tree estimators
        importances = np.mean([e.feature_importances_ for e in best_model.estimators_ if hasattr(e, 'feature_importances_')], axis=0)
    else:
        importances = np.zeros(len(available_features))

    importance_df = pd.DataFrame({
        'feature': available_features,
        'importance': importances
    }).sort_values('importance', ascending=False)
    logger.info(f"\nTop 10 features:")
    for _, row in importance_df.head(10).iterrows():
        logger.info(f"  {row['feature']}: {row['importance']:.4f}")

    # Save
    model_path = os.path.join(MODELS_DIR, f'best_model_{timeframe_label}.pkl')
    model_data = {
        'model': best_model,
        'model_type': model_type,
        'feature_columns': available_features,
        'timeframe': timeframe_label,
        'days': days,
        'f1_score': best_f1,
        'auc_score': best_auc,
    }
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
    logger.info(f"Saved: {model_path}")

    return model_path, model_type, best_f1, best_auc


def _objective_single(train_fn, trial, X, y, tscv):
    """Optuna objective for single model."""
    scores = []
    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        X_train_sm, y_train_sm = _apply_smote(X_train, y_train)

        model = train_fn(X_train_sm, y_train_sm, trial=trial)
        preds = model.predict(X_val)
        scores.append(f1_score(y_val, preds, zero_division=0))
    return np.mean(scores)


def train_all_models(stock_codes: list):
    """Train models for all timeframes."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    logger.info("=== Multi-Timeframe Model Training ===")
    logger.info(f"Ensemble={USE_ENSEMBLE}, Stacking={USE_STACKING}, SMOTE={USE_SMOTE}")

    results = {}
    for label, days in TIMEFRAMES.items():
        path, model_type, f1, auc = train_single_timeframe(stock_codes, label, days)
        results[label] = {'path': path, 'type': model_type, 'f1': f1, 'auc': auc}

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
