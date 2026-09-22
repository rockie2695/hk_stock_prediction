"""
Model training - Optuna hyperparameter tuning + Walk-Forward validation.
Supports Voting/Stacking/Blending ensemble of XGBoost, LightGBM, RandomForest, CatBoost.
Includes SMOTE for class imbalance and comprehensive metrics (F1, AUC, Precision, Recall).

Priority rules:
  - USE_STACKING=True forces ensemble mode (overrides USE_ENSEMBLE=False)
  - USE_BLENDING=True uses blending ensemble (stacking with out-of-fold predictions)
  - USE_ENSEMBLE=True + USE_STACKING=False + USE_BLENDING=False → VotingClassifier
  - USE_ENSEMBLE=False + USE_STACKING=False + USE_BLENDING=False → single best model
  - USE_SMOTE works with any of the above modes
  
Parallel training:
  - Timeframes (1d, 5d, 20d) are trained in parallel using ProcessPoolExecutor
  - Significantly faster on multi-core systems
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
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import STOCK_LIST, USE_ENSEMBLE, USE_STACKING, USE_SMOTE, USE_CATBOOST, USE_BLENDING, USE_GPU, USE_CLASS_WEIGHTS, USE_WALK_FORWARD
from src.data_fetcher import fetch_stock_data
from src.feature_engineering import compute_features, compute_target_days, compute_extended_features, FEATURE_COLUMNS, filter_correlated_features
from src.logger import setup_logger

logger = setup_logger('train_model')

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
CACHE_DIR = os.path.join(PROJECT_ROOT, 'cache')  # 共享數據暫存目錄 / Shared-data temp dir
os.makedirs(CACHE_DIR, exist_ok=True)

# Timeframes: label -> days ahead
TIMEFRAMES = {'1d': 1, '5d': 5, '20d': 20}

# Try importing CatBoost
try:
    import catboost as cb
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False
    logger.warning("CatBoost not installed. Install with: pip install catboost")


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
        hsi = yf.download('^HSI', start=start_date, end=end_date, progress=False, auto_adjust=False)
        if not hsi.empty:
            if isinstance(hsi.columns, pd.MultiIndex):
                hsi.columns = hsi.columns.get_level_values(0)
            data['hsi_close'] = hsi['Close']
            logger.info(f"  HSI: {len(hsi)} rows")
    except Exception as e:
        logger.warning(f"  HSI fetch failed: {e}")

    # USD/HKD
    try:
        fx = yf.download('HKD=X', start=start_date, end=end_date, progress=False, auto_adjust=False)
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
        market_df['hsi_ret_5d'] = market_df['hsi_close'].pct_change(5, fill_method=None)
        market_df['hsi_ret_20d'] = market_df['hsi_close'].pct_change(20, fill_method=None)
    if 'usdhkd' in market_df.columns:
        market_df['usdhkd_change'] = market_df['usdhkd'].pct_change(5, fill_method=None)

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
            
            # Compute extended features (sentiment, sector, short selling, etc.)
            try:
                df = compute_extended_features(df, code)
            except Exception as e:
                logger.warning(f"    Extended features failed for {code}: {e}")

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

    # Filter highly correlated features to reduce redundancy
    # Use 0.95 threshold to keep informative but correlated features
    # 使用 0.95 閾值保留有資訊量但相關的特徵
    available_features = filter_correlated_features(combined, available_features, threshold=0.95)

    return combined, available_features


def prepare_shared_data(stock_codes: list) -> tuple:
    """
    Prepare shared data for parallel training (fetched ONCE, reused by all timeframes).
    準備平行訓練共享數據 (只獲取一次，所有時間範圍重用)。
    
    This fetches OHLCV, computes features, extended features, and market context.
    Only the target variable (1d/5d/20d) differs per timeframe.
    
    Returns:
        Tuple of (combined_df, available_features, temp_file_path)
        The temp_file_path contains the serialized data for workers to load.
    """
    import tempfile
    
    logger.info("\n=== Preparing shared data (fetched ONCE for all timeframes) ===")
    logger.info("=== 準備共享數據 (所有時間範圍只獲取一次) ===")
    
    market_df = fetch_market_data()

    all_data = []
    for code in stock_codes:
        try:
            logger.info(f"  Fetching data for {code}...")
            df = fetch_stock_data(code, years=3)
            df = compute_features(df)
            df['stock_code'] = code
            
            # Compute extended features (sentiment, sector, short selling, etc.)
            # These are timeframe-independent
            try:
                df = compute_extended_features(df, code)
            except Exception as e:
                logger.warning(f"    Extended features failed for {code}: {e}")

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

    # Drop rows with NaN in features (but NOT target, since target doesn't exist yet)
    combined = combined.dropna(subset=available_features)

    # Filter highly correlated features
    available_features = filter_correlated_features(combined, available_features, threshold=0.95)
    
    logger.info(f"Shared data: {len(combined)} rows, {len(available_features)} features")
    
    # Save to temp file for workers to load
    temp_data = {
        'combined': combined,
        'available_features': available_features,
    }
    temp_path = os.path.join(CACHE_DIR, '_shared_training_data.pkl')
    with open(temp_path, 'wb') as f:
        pickle.dump(temp_data, f)
    logger.info(f"Shared data saved to: {temp_path}")
    
    return combined, available_features, temp_path


def prepare_data_with_shared(days: int, shared_data_path: str) -> pd.DataFrame:
    """
    Prepare data for a specific timeframe using pre-fetched shared data.
    使用預獲取的共享數據為特定時間範圍準備數據。
    
    Only computes the target variable (1d/5d/20d) — the expensive part is already done.
    只計算目標變數 (1d/5d/20d) — 昂貴的部分已完成。
    """
    import tempfile
    
    logger.info(f"  Loading shared data and computing {days}-day target...")
    
    # Load shared data
    with open(shared_data_path, 'rb') as f:
        temp_data = pickle.load(f)
    
    combined = temp_data['combined'].copy()
    available_features = temp_data['available_features']
    
    # Compute target for this timeframe
    combined = compute_target_days(combined, days)
    
    # Drop rows with NaN target
    combined = combined.dropna(subset=available_features + ['target'])
    
    logger.info(f"  {days}d data: {len(combined)} rows, target dist: {combined['target'].value_counts().to_dict()}")
    
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


def _scale_pos_weight(y) -> float:
    """Class-imbalance weight (capped at 3x). Only applied when USE_CLASS_WEIGHTS=True.
    類別不平衡權重 (上限 3 倍)。僅在 USE_CLASS_WEIGHTS=True 時套用。

    Note / 注意: When SMOTE is enabled the training set is already balanced,
    so this is usually a no-op. It matters when USE_SMOTE=False.
    當 SMOTE 啟用時訓練集已平衡，此權重通常無效。僅在 USE_SMOTE=False 時有意義。
    """
    if not USE_CLASS_WEIGHTS:
        return 1.0
    n0 = (y == 0).sum()
    n1 = (y == 1).sum()
    return min(n0 / n1, 3.0) if n1 > 0 else 1.0


def _purged_splits(tscv, X, days: int = None):
    """TimeSeriesSplit with purge/embargo to prevent label leakage.
    時間序列分割加上 purge/embargo 以防止標籤洩漏。

    Labels use shift(-days): sample i's label window spans i..i+days.
    Training samples whose window overlaps the validation fold leak future
    information into training. This drops them (purge) plus a 1-day embargo.
    標籤使用 shift(-days)：樣本 i 的標籤視窗跨越 i..i+days。
    訓練樣本若視窗與驗證折重疊會洩漏未來資訊，此處刪除 (purge) 並加 1 天 embargo。

    Args / 參數:
        tscv: TimeSeriesSplit cross-validator / 時間序列交叉驗證器
        X: Feature DataFrame (only length is used) / 特徵 DataFrame (僅使用長度)
        days: Target horizon in days (1, 5, or 20). None = no purge.
             / 目標天數。None = 不 purge。

    Returns / 返回:
        List of (train_idx, val_idx) tuples with purged training indices.
        / 帶 purge 訓練索引的 (train_idx, val_idx) 列表。
    """
    splits = []
    for train_idx, val_idx in tscv.split(X):
        if days is None or days <= 0:
            splits.append((train_idx, val_idx))
            continue
        val_start = val_idx[0]
        gap = days + 1  # purge window + 1-day embargo / purge 視窗 + 1 天 embargo
        purged = train_idx[train_idx + gap < val_start]
        if len(purged) < 60:
            # Too little data left to train — keep original fold (small dataset)
            # 剩餘數據太少無法訓練 — 保留原始折 (小數據集)
            logger.warning(f"    Purge left {len(purged)} train rows (<60), keeping original fold")
            splits.append((train_idx, val_idx))
        else:
            splits.append((purged, val_idx))
    return splits


def train_xgboost(X_train, y_train, trial=None, eval_set=None):
    """Train XGBoost with optional Optuna params."""
    scale_pos_weight = _scale_pos_weight(y_train)

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


def train_lightgbm(X_train, y_train, trial=None, eval_set=None):
    """Train LightGBM with optional Optuna params."""
    scale_pos_weight = _scale_pos_weight(y_train)

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


def train_random_forest(X_train, y_train, trial=None, eval_set=None):
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


def train_catboost(X_train, y_train, trial=None, eval_set=None):
    """Train CatBoost with optional Optuna params and early stopping."""
    if not HAS_CATBOOST:
        raise ImportError("CatBoost not installed")

    if trial:
        params = {
            'iterations': trial.suggest_int('cb_iterations', 200, 300),
            'depth': trial.suggest_int('cb_depth', 4, 6),
            'learning_rate': trial.suggest_float('cb_learning_rate', 0.05, 0.3, log=True),
            'l2_leaf_reg': trial.suggest_float('cb_l2_leaf_reg', 1e-8, 10.0, log=True),
        }
    else:
        params = {
            'iterations': 250,
            'depth': 6,
        }

    # Detect GPU availability
    task_type = _detect_gpu_task_type()

    # Early stopping: only when eval_set is provided
    fit_params = {}
    if eval_set is not None:
        fit_params['eval_set'] = eval_set
        fit_params['early_stopping_rounds'] = 30
        fit_params['verbose'] = False

    cb_kwargs = dict(
        random_seed=42,
        logging_level='Silent',
        allow_writing_files=False,
        task_type=task_type,
        thread_count=4,  # Limit CPU threads to reduce RAM
        border_count=128,  # Limit histogram bins
        max_ctr_complexity=2,  # Reduce memory for categorical features
    )
    if USE_CLASS_WEIGHTS:
        cb_kwargs['auto_class_weights'] = 'Balanced'

    model = cb.CatBoostClassifier(**params, **cb_kwargs)
    model.fit(X_train, y_train, **fit_params)
    return model


def _detect_gpu_task_type() -> str:
    """Detect if GPU is available for CatBoost training."""
    if not USE_GPU:
        logger.info("  GPU disabled (USE_GPU=False) - using CPU for CatBoost")
        return 'CPU'
    
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi'], capture_output=True, timeout=5)
        if result.returncode == 0:
            logger.info("  GPU detected (NVIDIA) - using GPU for CatBoost")
            return 'GPU'
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    
    logger.info("  No GPU detected - using CPU for CatBoost")
    return 'CPU'


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


def _get_estimator_list(xgb_model, lgb_model, rf_model, cb_model=None):
    """Build estimator list based on enabled models."""
    estimators = [('xgb', xgb_model), ('lgb', lgb_model), ('rf', rf_model)]
    if USE_CATBOOST and cb_model is not None:
        estimators.append(('cb', cb_model))
    return estimators


def objective_ensemble(trial, X, y, tscv, days: int = None):
    """Optuna objective for ensemble: tune individual model params + voting weights.
    
    days: Target horizon for purge/embargo (None = no purge) / 目標天數用於 purge/embargo
    """
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
    
    n0 = (y == 0).sum()
    n1 = (y == 1).sum()
    scale_pos_weight = _scale_pos_weight(y)

    xgb_model = xgb.XGBClassifier(**xgb_params, scale_pos_weight=scale_pos_weight, random_state=42, use_label_encoder=False, eval_metric='logloss', verbosity=0)
    lgb_model = lgb.LGBMClassifier(**lgb_params, scale_pos_weight=scale_pos_weight, random_state=42, verbosity=-1)
    rf_model = RandomForestClassifier(**rf_params, random_state=42, n_jobs=-1)
    
    # Tune CatBoost if enabled
    cb_model = None
    if USE_CATBOOST and HAS_CATBOOST:
        cb_params = {
            'iterations': trial.suggest_int('cb_iterations', 200, 300),
            'depth': trial.suggest_int('cb_depth', 4, 6),
            'learning_rate': trial.suggest_float('cb_learning_rate', 0.05, 0.3, log=True),
        }
        task_type = _detect_gpu_task_type()
        cb_kwargs = dict(
            random_seed=42, logging_level='Silent', allow_writing_files=False,
            task_type=task_type, thread_count=4, border_count=128, max_ctr_complexity=2,
        )
        if USE_CLASS_WEIGHTS:
            cb_kwargs['auto_class_weights'] = 'Balanced'
        cb_model = cb.CatBoostClassifier(**cb_params, **cb_kwargs)

    # Tune voting weights
    w1 = trial.suggest_float('w_xgb', 0.1, 2.0)
    w2 = trial.suggest_float('w_lgb', 0.1, 2.0)
    w3 = trial.suggest_float('w_rf', 0.1, 2.0)
    w4 = trial.suggest_float('w_cb', 0.1, 2.0) if USE_CATBOOST and HAS_CATBOOST else 1.0

    estimator_list = _get_estimator_list(xgb_model, lgb_model, rf_model, cb_model)
    
    if USE_BLENDING:
        # Blending: use out-of-fold predictions
        ensemble = _create_blending_ensemble(estimator_list, tscv, days=days)
    elif USE_STACKING:
        ensemble = StackingClassifier(
            estimators=estimator_list,
            final_estimator=LogisticRegression(random_state=42),
            cv=3,
            passthrough=False
        )
    else:
        weights = [w1, w2, w3]
        if USE_CATBOOST and HAS_CATBOOST:
            weights.append(w4)
        ensemble = VotingClassifier(
            estimators=estimator_list,
            voting='soft',
            weights=weights
        )

    scores = []
    for train_idx, val_idx in _purged_splits(tscv, X, days):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        X_train_sm, y_train_sm = _apply_smote(X_train, y_train)

        ensemble.fit(X_train_sm, y_train_sm)
        preds = ensemble.predict(X_val)
        scores.append(f1_score(y_val, preds, zero_division=0))
    return np.mean(scores)


def _create_blending_ensemble(estimators, tscv, days: int = None):
    """Create a blending ensemble using out-of-fold predictions.
    
    days: Target horizon for purge/embargo on OOF folds (None = no purge)
         / 目標天數用於 OOF 折的 purge/embargo
    """
    class BlendingClassifier:
        def __init__(self, estimators, tscv, days=None):
            self.estimators = estimators
            self.tscv = tscv
            self.days = days
            self.meta_model = LogisticRegression(random_state=42)
            self.fitted_estimators = []
            
        def fit(self, X, y):
            # Generate out-of-fold predictions for meta-features
            # (purged to prevent label leakage from overlapping windows)
            # (purge 以防止重疊視窗造成的標籤洩漏)
            oof_predictions = np.zeros((len(X), len(self.estimators)))
            
            for fold_idx, (train_idx, val_idx) in enumerate(_purged_splits(self.tscv, X, self.days)):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train = y.iloc[train_idx]
                
                for est_idx, (name, estimator) in enumerate(self.estimators):
                    # Clone and fit estimator
                    from sklearn.base import clone
                    est_clone = clone(estimator)
                    est_clone.fit(X_train, y_train)
                    oof_predictions[val_idx, est_idx] = est_clone.predict_proba(X_val)[:, 1]
            
            # Train meta-model on OOF predictions
            self.meta_model.fit(oof_predictions, y)
            
            # Fit all estimators on full data for final predictions
            self.fitted_estimators = []
            for name, estimator in self.estimators:
                from sklearn.base import clone
                est_clone = clone(estimator)
                est_clone.fit(X, y)
                self.fitted_estimators.append((name, est_clone))
            
            return self
        
        def predict(self, X):
            meta_features = self._get_meta_features(X)
            return self.meta_model.predict(meta_features)
        
        def predict_proba(self, X):
            meta_features = self._get_meta_features(X)
            return self.meta_model.predict_proba(meta_features)
        
        def _get_meta_features(self, X):
            meta_features = np.zeros((len(X), len(self.fitted_estimators)))
            for est_idx, (name, estimator) in enumerate(self.fitted_estimators):
                meta_features[:, est_idx] = estimator.predict_proba(X)[:, 1]
            return meta_features
    
    return BlendingClassifier(estimators, tscv, days=days)


def train_single_timeframe(stock_codes: list, timeframe_label: str, days: int, shared_data_path: str = None):
    """Train and save model for one timeframe.
    
    Args:
        stock_codes: List of stock codes
        timeframe_label: '1d', '5d', or '20d'
        days: Number of days ahead for target
        shared_data_path: Path to pre-fetched shared data (optional). If provided, 
                         skips data fetching and only computes target.
    """
    # Resolve training mode: USE_STACKING or USE_BLENDING overrides USE_ENSEMBLE
    use_ensemble = USE_ENSEMBLE or USE_STACKING or USE_BLENDING

    logger.info(f"\n{'='*50}")
    logger.info(f"Training model for {timeframe_label} ({days}-day ahead)")
    logger.info(f"Ensemble: {use_ensemble}, Stacking: {USE_STACKING}, Blending: {USE_BLENDING}, SMOTE: {USE_SMOTE}")
    logger.info(f"CatBoost: {USE_CATBOOST and HAS_CATBOOST}")
    logger.info(f"{'='*50}")

    # Use shared data if available (fast path), otherwise fetch fresh (fallback)
    if shared_data_path and os.path.exists(shared_data_path):
        data, available_features = prepare_data_with_shared(days, shared_data_path)
    else:
        data, available_features = prepare_data(stock_codes, days)

    X = data[available_features]
    y = data['target']

    logger.info(f"Dataset: {len(X)} rows, {len(available_features)} features")
    logger.info(f"Target distribution: {y.value_counts().to_dict()}")

    tscv = TimeSeriesSplit(n_splits=5)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    if use_ensemble:
        # Train ensemble
        logger.info("Optuna Ensemble (50 trials)...")
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: objective_ensemble(trial, X, y, tscv, days), n_trials=50)
        logger.info(f"  Ensemble best F1 (CV): {study.best_value:.4f}")

        # Retrain on LAST fold (with purge/embargo)
        splits = _purged_splits(tscv, X, days)
        train_idx, val_idx = splits[-1]
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        X_train_sm, y_train_sm = _apply_smote(X_train, y_train)

        best_p = study.best_params
        n0 = (y_train_sm == 0).sum()
        n1 = (y_train_sm == 1).sum()
        scale_pos_weight = _scale_pos_weight(y_train_sm)

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
        
        cb_model = None
        if USE_CATBOOST and HAS_CATBOOST:
            cb_kwargs = dict(
                iterations=best_p.get('cb_iterations', 250),
                depth=best_p.get('cb_depth', 6),
                learning_rate=best_p.get('cb_learning_rate', 0.1),
                random_seed=42, verbose=0,
            )
            if USE_CLASS_WEIGHTS:
                cb_kwargs['auto_class_weights'] = 'Balanced'
            cb_model = cb.CatBoostClassifier(**cb_kwargs)

        estimator_list = _get_estimator_list(xgb_model, lgb_model, rf_model, cb_model)
        
        if USE_BLENDING:
            ensemble = _create_blending_ensemble(estimator_list, tscv, days=days)
            model_type = 'blending'
        elif USE_STACKING:
            ensemble = StackingClassifier(
                estimators=estimator_list,
                final_estimator=LogisticRegression(random_state=42), cv=3, passthrough=False)
            model_type = 'stacking'
        else:
            w1, w2, w3 = best_p['w_xgb'], best_p['w_lgb'], best_p['w_rf']
            weights = [w1, w2, w3]
            if USE_CATBOOST and HAS_CATBOOST:
                weights.append(best_p.get('w_cb', 1.0))
            ensemble = VotingClassifier(
                estimators=estimator_list,
                voting='soft', weights=weights)
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
        # Single model selection - train all and compare
        logger.info("Training individual models for comparison...")
        
        results = {}
        
        # XGBoost
        logger.info("Optuna XGBoost (50 trials)...")
        study_xgb = optuna.create_study(direction='maximize')
        study_xgb.optimize(lambda trial: _objective_single(train_xgboost, trial, X, y, tscv, days), n_trials=50)
        results['xgboost'] = study_xgb.best_value
        logger.info(f"  XGBoost best F1: {study_xgb.best_value:.4f}")

        # LightGBM
        logger.info("Optuna LightGBM (50 trials)...")
        study_lgb = optuna.create_study(direction='maximize')
        study_lgb.optimize(lambda trial: _objective_single(train_lightgbm, trial, X, y, tscv, days), n_trials=50)
        results['lightgbm'] = study_lgb.best_value
        logger.info(f"  LightGBM best F1: {study_lgb.best_value:.4f}")
        
        # CatBoost (if enabled)
        if USE_CATBOOST and HAS_CATBOOST:
            logger.info("Optuna CatBoost (50 trials)...")
            study_cb = optuna.create_study(direction='maximize')
            study_cb.optimize(lambda trial: _objective_single(train_catboost, trial, X, y, tscv, days), n_trials=50)
            results['catboost'] = study_cb.best_value
            logger.info(f"  CatBoost best F1: {study_cb.best_value:.4f}")

        # Select winner
        best_model_name = max(results, key=results.get)
        logger.info(f"\n{'='*50}")
        logger.info("MODEL COMPARISON")
        logger.info(f"{'='*50}")
        for name, f1 in sorted(results.items(), key=lambda x: x[1], reverse=True):
            marker = " ← WINNER" if name == best_model_name else ""
            logger.info(f"  {name:12s}: F1={f1:.4f}{marker}")
        logger.info(f"{'='*50}")

        splits = list(tscv.split(X))
        train_idx, val_idx = splits[-1]
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        X_train_sm, y_train_sm = _apply_smote(X_train, y_train)

        # Train winner
        if best_model_name == 'xgboost':
            best_xgb_params = {k.replace('xgb_', ''): v for k, v in study_xgb.best_params.items()}
            best_model = train_xgboost(X_train_sm, y_train_sm, trial=None)
            best_model.set_params(**best_xgb_params)
            best_model.fit(X_train_sm, y_train_sm)
        elif best_model_name == 'lightgbm':
            best_lgb_params = {k.replace('lgb_', ''): v for k, v in study_lgb.best_params.items()}
            best_model = train_lightgbm(X_train_sm, y_train_sm, trial=None)
            best_model.set_params(**best_lgb_params)
            best_model.fit(X_train_sm, y_train_sm)
        elif best_model_name == 'catboost':
            best_cb_params = {k.replace('cb_', ''): v for k, v in study_cb.best_params.items()}
            best_model = train_catboost(X_train_sm, y_train_sm, trial=None, eval_set=(X_val, y_val))
            best_model.set_params(**best_cb_params)
            best_model.fit(X_train_sm, y_train_sm)

        preds = best_model.predict(X_val)
        proba = best_model.predict_proba(X_val)[:, 1]
        best_f1 = f1_score(y_val, preds, zero_division=0)
        best_auc = roc_auc_score(y_val, proba)
        precision = precision_score(y_val, preds, zero_division=0)
        recall = recall_score(y_val, preds, zero_division=0)
        model_type = best_model_name

        _save_roc_curve(y_val, proba, timeframe_label, model_type)

    logger.info(f"Winner: {model_type}")
    logger.info(f"  F1={best_f1:.4f}, AUC={best_auc:.4f}, Precision={precision:.4f}, Recall={recall:.4f}")

    # Feature importance
    if hasattr(best_model, 'feature_importances_'):
        importances = best_model.feature_importances_
    elif hasattr(best_model, 'estimators_'):
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

    # Save feature importance to CSV
    importance_path = os.path.join(MODELS_DIR, f'feature_importance_{timeframe_label}.csv')
    importance_df.to_csv(importance_path, index=False)
    logger.info(f"Feature importance saved: {importance_path}")

    # Optimize Buy/Sell thresholds on validation set
    best_proba = best_model.predict_proba(X_val)[:, 1]
    best_thresh_buy, best_thresh_sell, best_thresh_f1 = _optimize_thresholds(y_val, best_proba)
    logger.info(f"  Optimized thresholds: Buy>{best_thresh_buy:.3f}, Sell<{best_thresh_sell:.3f} (F1={best_thresh_f1:.4f})")

    # Save model with metadata + thresholds
    model_data = {
        'model': best_model,
        'model_type': model_type,
        'feature_columns': available_features,
        'timeframe': timeframe_label,
        'days': days,
        'f1_score': best_f1,
        'auc_score': best_auc,
        'threshold_buy': best_thresh_buy,
        'threshold_sell': best_thresh_sell,
    }

    # Model versioning: save timestamped version, keep last 5
    from datetime import datetime as dt
    import pytz
    timestamp = dt.now(pytz.timezone('Asia/Hong_Kong')).strftime('%Y%m%d_%H%M%S')
    versioned_path = os.path.join(MODELS_DIR, f'best_model_{timeframe_label}_{timestamp}.pkl')
    with open(versioned_path, 'wb') as f:
        pickle.dump(model_data, f)
    logger.info(f"Versioned model saved: {versioned_path}")

    # Always overwrite the "current" model file (used by predict_upload.py)
    model_path = os.path.join(MODELS_DIR, f'best_model_{timeframe_label}.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
    logger.info(f"Current model saved: {model_path}")

    # Cleanup: keep only last 5 versioned models per timeframe
    _cleanup_old_models(timeframe_label, keep=5)
    
    # Save tree visualizations / 儲存樹狀圖
    _save_tree_visualizations(best_model, timeframe_label, model_type, available_features)
    
    # Walk-forward backtest (training-time, honest out-of-sample estimate)
    # 走動前推回測 (訓練時，誠實的樣本外估計)
    if USE_WALK_FORWARD:
        logger.info("Running walk-forward backtest... / 執行走動前推回測...")
        _walk_forward_backtest(data, available_features, days, timeframe_label)

    return model_path, model_type, best_f1, best_auc


def _optimize_thresholds(y_true, y_proba):
    """Find optimal Buy/Sell thresholds that maximize F1 on validation set."""
    from sklearn.metrics import f1_score
    best_f1 = 0
    best_buy = 0.55
    best_sell = 0.45
    for buy_t in np.arange(0.50, 0.70, 0.01):
        for sell_t in np.arange(0.30, 0.50, 0.01):
            if sell_t >= buy_t:
                continue
            preds = np.where(y_proba > buy_t, 1, np.where(y_proba < sell_t, 0, -1))
            # Only evaluate on non-Hold predictions
            mask = preds != -1
            if mask.sum() < 10:
                continue
            f1 = f1_score(y_true[mask], preds[mask], zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_buy = buy_t
                best_sell = sell_t
    return best_buy, best_sell, best_f1


def _cleanup_old_models(timeframe_label: str, keep: int = 5):
    """Keep only the last N versioned model files, delete older ones."""
    import glob
    pattern = os.path.join(MODELS_DIR, f'best_model_{timeframe_label}_*.pkl')
    files = sorted(glob.glob(pattern))
    if len(files) > keep:
        for f in files[:-keep]:
            try:
                os.remove(f)
                logger.info(f"  Cleaned up old model: {os.path.basename(f)}")
            except Exception as e:
                logger.warning(f"  Failed to remove {f}: {e}")


def _save_tree_visualizations(model, timeframe_label: str, model_type: str, feature_names: list):
    """
    Export model visualization images for VotingClassifier ensemble.
    為 VotingClassifier 集成模型匯出模型可視化影像。

    - RandomForest: actual tree structure (sklearn plot_tree, pure matplotlib)
    - XGBoost / LightGBM: feature importance bar chart (pure matplotlib, no Graphviz)
    - CatBoost: feature importance bar chart (pure matplotlib, no Graphviz)

    Only exports for voting/stacking/blending ensemble models.
    Saves PNG files to models/trees/ directory:
    - tree_{timeframe}_{model_name}.png
    """
    trees_dir = os.path.join(MODELS_DIR, 'trees')
    os.makedirs(trees_dir, exist_ok=True)
    
    # Only export trees for ensemble models (voting, stacking, blending)
    # 只為集成模型 (voting, stacking, blending) 匯出樹狀圖
    if model_type not in ('voting', 'stacking', 'blending'):
        logger.info(f"  Skipping tree export for single model type: {model_type}")
        return
    
    # For voting/stacking: extract sub-estimators from the fitted ensemble
    # 對於 voting/stacking：從擬合的集成模型中提取子估計器
    estimators_to_plot = []
    
    if hasattr(model, 'named_estimators_'):
        # VotingClassifier or StackingClassifier
        estimators_to_plot = list(model.named_estimators_.items())
    elif hasattr(model, 'fitted_estimators'):
        # Blending ensemble
        estimators_to_plot = [(name, est) for name, est in model.fitted_estimators]
    else:
        logger.warning("  Model has no estimators to plot")
        return
    
    saved_count = 0
    
    for est_name, estimator in estimators_to_plot:
        tree_path = os.path.join(trees_dir, f'tree_{timeframe_label}_{est_name}.png')
        
        try:
            # XGBoost — feature importance (pure matplotlib, no Graphviz)
            if isinstance(estimator, xgb.XGBClassifier):
                try:
                    fig, ax = plt.subplots(figsize=(16, 10))
                    xgb.plot_importance(estimator, ax=ax, max_num_features=20,
                                        importance_type='gain', title='')
                    ax.set_title(f'XGBoost Feature Importance - {timeframe_label}', fontsize=14)
                    fig.tight_layout()
                    fig.savefig(tree_path, dpi=300, bbox_inches='tight')
                    plt.close(fig)
                    saved_count += 1
                    logger.info(f"  Tree saved: {os.path.basename(tree_path)}")
                except Exception as e:
                    plt.close('all')
                    logger.warning(f"  XGBoost tree export failed: {e}")

            # LightGBM — feature importance (pure matplotlib, no Graphviz)
            elif isinstance(estimator, lgb.LGBMClassifier):
                try:
                    fig, ax = plt.subplots(figsize=(16, 10))
                    lgb.plot_importance(estimator, ax=ax, max_num_features=20,
                                        importance_type='gain', title='')
                    ax.set_title(f'LightGBM Feature Importance - {timeframe_label}', fontsize=14)
                    fig.tight_layout()
                    fig.savefig(tree_path, dpi=300, bbox_inches='tight')
                    plt.close(fig)
                    saved_count += 1
                    logger.info(f"  Tree saved: {os.path.basename(tree_path)}")
                except Exception as e:
                    plt.close('all')
                    logger.warning(f"  LightGBM tree export failed: {e}")

            # RandomForest — actual tree structure (sklearn plot_tree, pure matplotlib)
            elif isinstance(estimator, RandomForestClassifier):
                try:
                    from sklearn.tree import plot_tree
                    fig, ax = plt.subplots(figsize=(24, 12))
                    plot_tree(estimator.estimators_[0],
                              feature_names=feature_names[:len(estimator.feature_importances_)],
                              filled=True, rounded=True, max_depth=3, ax=ax)
                    ax.set_title(f'RandomForest Tree - {timeframe_label}')
                    fig.tight_layout()
                    fig.savefig(tree_path, dpi=300, bbox_inches='tight')
                    plt.close(fig)
                    saved_count += 1
                    logger.info(f"  Tree saved: {os.path.basename(tree_path)}")
                except Exception as e:
                    plt.close('all')
                    logger.warning(f"  RandomForest tree export failed: {e}")

            # CatBoost — feature importance bar chart (pure matplotlib, no Graphviz)
            elif HAS_CATBOOST and isinstance(estimator, cb.CatBoostClassifier):
                try:
                    importances = estimator.get_feature_importance()
                    names = list(estimator.feature_names_) if estimator.feature_names_ else feature_names[:len(importances)]
                    # Top 20 features
                    idx = np.argsort(importances)[-20:]
                    fig, ax = plt.subplots(figsize=(16, 10))
                    ax.barh([names[i] for i in idx], importances[idx], color='#2ca02c')
                    ax.set_xlabel('Importance (gain)')
                    ax.set_title(f'CatBoost Feature Importance - {timeframe_label}', fontsize=14)
                    fig.tight_layout()
                    fig.savefig(tree_path, dpi=300, bbox_inches='tight')
                    plt.close(fig)
                    saved_count += 1
                    logger.info(f"  Tree saved: {os.path.basename(tree_path)}")
                except Exception as e:
                    plt.close('all')
                    logger.warning(f"  CatBoost tree export failed: {e}")

        except Exception as e:
            logger.warning(f"  Failed to save tree for {est_name}: {e}")
            continue
    
    if saved_count > 0:
        logger.info(f"  Total trees saved: {saved_count}/{len(estimators_to_plot)}")
    else:
        logger.warning("  No tree visualizations could be saved")


def _objective_single(train_fn, trial, X, y, tscv, days: int = None):
    """
    Optuna objective function for single model hyperparameter tuning.
    
    Args:
        train_fn: Training function (train_xgboost, train_lightgbm, or train_catboost)
        trial: Optuna trial object for suggesting hyperparameters
        X: Feature DataFrame
        y: Target Series
        tscv: TimeSeriesSplit cross-validator
        days: Target horizon for purge/embargo (None = no purge)
             / 目標天數用於 purge/embargo
        
    Returns:
        Mean F1 score across CV folds
    """
    scores = []
    for train_idx, val_idx in _purged_splits(tscv, X, days):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        X_train_sm, y_train_sm = _apply_smote(X_train, y_train)

        model = train_fn(X_train_sm, y_train_sm, trial=trial, eval_set=(X_val, y_val))
        preds = model.predict(X_val)
        scores.append(f1_score(y_val, preds, zero_division=0))
    return np.mean(scores)


def _train_timeframe_worker(args):
    """Worker function for parallel training of a single timeframe."""
    stock_codes, label, days, shared_data_path = args
    try:
        return train_single_timeframe(stock_codes, label, days, shared_data_path)
    except Exception as e:
        logger.error(f"Training failed for {label}: {e}")
        return None


def _build_default_voting(X_train, y_train):
    """Build a lightweight voting ensemble with default hyperparameters (no Optuna).
    使用預設超參數建立輕量投票集成 (無 Optuna)。Used for fast walk-forward backtest.
    """
    scale_pos_weight = _scale_pos_weight(y_train)
    ests = []
    ests.append(('xgb', xgb.XGBClassifier(
        n_estimators=150, max_depth=6, learning_rate=0.1,
        scale_pos_weight=scale_pos_weight, random_state=42,
        use_label_encoder=False, eval_metric='logloss', verbosity=0)))
    ests.append(('lgb', lgb.LGBMClassifier(
        n_estimators=150, max_depth=6, learning_rate=0.1,
        scale_pos_weight=scale_pos_weight, random_state=42, verbosity=-1)))
    ests.append(('rf', RandomForestClassifier(
        n_estimators=150, max_depth=10, random_state=42, n_jobs=-1)))
    if USE_CATBOOST and HAS_CATBOOST:
        cb_kwargs = dict(
            iterations=200, depth=6, learning_rate=0.1, random_seed=42,
            logging_level='Silent', allow_writing_files=False,
            task_type=_detect_gpu_task_type(), thread_count=4,
            border_count=128, max_ctr_complexity=2,
        )
        if USE_CLASS_WEIGHTS:
            cb_kwargs['auto_class_weights'] = 'Balanced'
        ests.append(('cb', cb.CatBoostClassifier(**cb_kwargs)))
    model = VotingClassifier(estimators=ests, voting='soft', weights=[1.0] * len(ests))
    model.fit(X_train, y_train)
    return model


def _walk_forward_backtest(data, available_features, days, timeframe_label):
    """Walk-forward backtest: train on expanding purged windows, predict forward, simulate.
    走動前推回測：在擴展 purge 視窗上訓練，向前預測並模擬。

    Uses default-parameter ensembles (no Optuna) for speed. Saves a CSV + plot.
    使用預設參數集成 (無 Optuna) 以加快速度。儲存 CSV 與圖表。

    Returns / 返回:
        Dict of metrics, or None on failure.
    """
    try:
        X = data[available_features]
        y = data['target']
        tscv = TimeSeriesSplit(n_splits=5)
        splits = _purged_splits(tscv, X, days)

        results = []
        for train_idx, val_idx in splits:
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train = y.iloc[train_idx]
            X_train_sm, y_train_sm = _apply_smote(X_train, y_train)
            model = _build_default_voting(X_train_sm, y_train_sm)
            proba = model.predict_proba(X_val)[:, 1]
            for gi, p in zip(val_idx, proba):
                results.append((int(gi), float(p)))
        if not results:
            return None
        results.sort(key=lambda r: r[0])
        gidxs = np.array([r[0] for r in results])
        all_proba = np.array([r[1] for r in results])
        all_y = y.iloc[gidxs].values

        # Forward returns over the horizon using Close / 使用收盤價計算期間前向報酬
        stats = {}
        if 'Close' in data.columns:
            close = data['Close'].reset_index(drop=True)
            fwd = close.shift(-days) / close - 1
            ret_fwd = fwd.iloc[gidxs].values
            signal = np.where(all_proba > 0.55, 1, np.where(all_proba < 0.45, -1, 0))
            strat_ret = np.mean(ret_fwd[signal == 1]) if (signal == 1).any() else 0.0
            hold_ret = np.mean(ret_fwd) if len(ret_fwd) else 0.0
            stats['strategy_return'] = round(float(strat_ret), 6)
            stats['buy_hold_return'] = round(float(hold_ret), 6)
            stats['alpha'] = round(float(strat_ret - hold_ret), 6)
            stats['buy_ratio'] = round(float((signal == 1).mean()), 4)
            stats['sell_ratio'] = round(float((signal == -1).mean()), 4)

        f1 = f1_score(all_y, (all_proba > 0.5).astype(int), zero_division=0)
        auc = roc_auc_score(all_y, all_proba)
        stats['f1'] = round(float(f1), 4)
        stats['auc'] = round(float(auc), 4)
        stats['samples'] = int(len(all_y))

        # Save CSV / 儲存 CSV
        csv_path = os.path.join(MODELS_DIR, f'walk_forward_{timeframe_label}.csv')
        pd.DataFrame({'index': gidxs, 'proba': all_proba, 'target': all_y}).to_csv(csv_path, index=False)
        logger.info(f"  Walk-forward results saved: {csv_path}")

        # Plot / 繪圖
        try:
            plt.figure(figsize=(10, 5))
            plt.plot(np.arange(len(all_proba)), all_proba, 'b-', label='P(Buy)', linewidth=0.8)
            plt.axhline(0.55, color='g', linestyle='--', label='Buy threshold')
            plt.axhline(0.45, color='r', linestyle='--', label='Sell threshold')
            plt.title(f'Walk-Forward Backtest - {timeframe_label} (AUC={stats["auc"]})')
            plt.xlabel('Validation sample (time-ordered)')
            plt.ylabel('Probability')
            plt.legend()
            png_path = os.path.join(MODELS_DIR, f'walk_forward_{timeframe_label}.png')
            plt.savefig(png_path, dpi=100)
            plt.close()
            logger.info(f"  Walk-forward plot saved: {png_path}")
        except Exception as e:
            logger.warning(f"  Walk-forward plot failed: {e}")

        logger.info(f"  Walk-forward {timeframe_label}: F1={stats['f1']}, AUC={stats['auc']}, "
                    f"strategy_ret={stats.get('strategy_return')}, hold_ret={stats.get('buy_hold_return')}")
        return stats
    except Exception as e:
        logger.warning(f"  Walk-forward backtest failed for {timeframe_label}: {e}")
        return None


def train_all_models(stock_codes: list):
    """Train models for all timeframes in parallel.
    
    Optimization: Fetches data ONCE and shares across all 3 timeframes.
    每個時間範圍平行訓練。優化：只獲取一次數據，所有時間範圍共享。
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    logger.info("=== Multi-Timeframe Model Training ===")
    resolved_ensemble = USE_ENSEMBLE or USE_STACKING or USE_BLENDING
    logger.info(f"Ensemble={resolved_ensemble} (raw={USE_ENSEMBLE}), Stacking={USE_STACKING}, Blending={USE_BLENDING}, SMOTE={USE_SMOTE}")
    logger.info(f"CatBoost={USE_CATBOOST and HAS_CATBOOST}")
    
    # Step 1: Fetch shared data ONCE (saves ~60-70% of data processing time)
    # 步驟 1：只獲取一次共享數據 (節省 ~60-70% 數據處理時間)
    try:
        _, _, shared_data_path = prepare_shared_data(stock_codes)
    except Exception as e:
        logger.error(f"Failed to prepare shared data: {e}")
        logger.info("Falling back to per-worker data fetching...")
        shared_data_path = None

    # Prepare tasks (now include shared_data_path)
    tasks = [(stock_codes, label, days, shared_data_path) for label, days in TIMEFRAMES.items()]
    
    # Parallel training
    results = {}
    logger.info(f"\nTraining {len(tasks)} timeframes in parallel...")

    # try/finally guarantees temp file cleanup even if training crashes mid-run
    # try/finally 確保即使訓練中途崩潰也會清理暫存檔
    try:
        with ProcessPoolExecutor(max_workers=min(len(tasks), os.cpu_count() or 1)) as executor:
            future_to_label = {
                executor.submit(_train_timeframe_worker, task): task[1] 
                for task in tasks
            }
            
            for future in as_completed(future_to_label):
                label = future_to_label[future]
                try:
                    result = future.result()
                    if result:
                        path, model_type, f1, auc = result
                        results[label] = {'path': path, 'type': model_type, 'f1': f1, 'auc': auc}
                except Exception as e:
                    logger.error(f"Training failed for {label}: {e}")
    finally:
        # Cleanup shared data file (always, even on crash)
        if shared_data_path and os.path.exists(shared_data_path):
            try:
                os.remove(shared_data_path)
                logger.info("Cleaned up shared data file")
            except Exception:
                pass

    # Summary
    logger.info("\n" + "="*60)
    logger.info("TRAINING SUMMARY")
    logger.info("="*60)
    logger.info(f"{'Timeframe':<12} {'Model Type':<12} {'F1 Score':<12} {'AUC Score':<12}")
    logger.info("-"*60)
    for label in ['1d', '5d', '20d']:
        if label in results:
            r = results[label]
            logger.info(f"{label:<12} {r['type']:<12} {r['f1']:<12.4f} {r['auc']:<12.4f}")
    logger.info("="*60)


if __name__ == '__main__':
    logger.info("=== Model Training Started ===")
    logger.info(f"Stock codes: {STOCK_LIST}")
    try:
        train_all_models(STOCK_LIST)
        logger.info("All models trained successfully.")
        from src.notifier import notify_success
        notify_success("模型訓練 / Training", f"Stocks: {STOCK_LIST} / 股票: {STOCK_LIST}")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        from src.notifier import notify_failure
        notify_failure("訓練 / Training", e)
        sys.exit(1)
