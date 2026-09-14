"""
Online Learning - Incremental Model Updates / 增量學習 - 模型增量更新
Wraps models with warm-start capability for daily incremental updates.
為模型提供熱啟動功能，用於每日增量更新。

Instead of full retraining every day, this module allows incremental updates
using recent data, saving time and computational resources.
此模組允許使用近期數據進行增量更新，而非每天全量重訓，節省時間和計算資源。
"""
import os
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
import pytz
from src.logger import setup_logger

logger = setup_logger('online_learner')

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
ONLINE_DIR = os.path.join(MODELS_DIR, 'online')  # online model versions / 線上模型版本
os.makedirs(ONLINE_DIR, exist_ok=True)

HK_TZ = pytz.timezone('Asia/Hong_Kong')


def should_use_online_learning(timeframe_label: str) -> bool:
    """
    Decide whether to use online learning or full retrain.
    決定使用增量學習還是全量重訓。
    Criteria: last full train was > 7 days ago.
    條件：上次全量訓練距今超過 7 天。
    """
    model_path = os.path.join(MODELS_DIR, f'best_model_{timeframe_label}.pkl')
    if not os.path.exists(model_path):
        return False
    
    mtime = os.path.getmtime(model_path)
    age_days = (datetime.now().timestamp() - mtime) / (24 * 3600)
    
    return age_days > 7


def online_update(stock_codes: list, timeframe_label: str, days: int) -> bool:
    """
    Perform incremental model update using recent data.
    使用近期數據執行模型增量更新。
    
    Args / 參數:
        stock_codes: List of stock codes / 股票代碼列表
        timeframe_label: '1d', '5d', or '20d' / 時間範圍標籤
        days: Number of days ahead for target / 目標天數
        
    Returns / 返回:
        True if update was successful / 更新成功返回 True
    """
    try:
        import xgboost as xgb
        import lightgbm as lgb
        
        # Load current model / 載入當前模型
        model_path = os.path.join(MODELS_DIR, f'best_model_{timeframe_label}.pkl')
        if not os.path.exists(model_path):
            logger.warning(f"No model found for {timeframe_label} / 未找到 {timeframe_label} 模型")
            return False
        
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        model = model_data['model']
        feature_columns = model_data.get('feature_columns', [])
        
        # Fetch recent data (last 30 days) / 獲取近期數據 (最近30天)
        from src.data_fetcher import fetch_stock_data
        from src.feature_engineering import compute_features, compute_target_days
        
        recent_data = []
        for code in stock_codes:
            try:
                df = fetch_stock_data(code, years=1)
                df = compute_features(df)
                df = compute_target_days(df, days)
                df['stock_code'] = code
                recent_data.append(df.tail(60))  # last 60 days / 最近60天
            except Exception as e:
                logger.warning(f"Failed to fetch data for {code}: {e} / 無法獲取 {code} 數據")
                continue
        
        if not recent_data:
            return False
        
        combined = pd.concat(recent_data, ignore_index=True)
        available_features = [f for f in feature_columns if f in combined.columns]
        combined = combined.dropna(subset=available_features + ['target'])
        
        if len(combined) < 30:
            logger.warning("Not enough data for online update / 數據不足，無法增量更新")
            return False
        
        X = combined[available_features]
        y = combined['target']
        
        # Incremental update based on model type / 根據模型類型進行增量更新
        model_type = model_data.get('model_type', '')
        
        if 'xgb' in model_type or hasattr(model, 'get_xgb_weight'):
            # XGBoost warm start / XGBoost 熱啟動
            try:
                model.set_params(n_estimators=model.n_estimators + 50)
                model.fit(X, y, xgb_model=model.get_booster())
                logger.info(f"[online] {timeframe_label}: XGBoost updated with {len(X)} samples / XGBoost 已更新 {len(X)} 個樣本")
            except Exception as e:
                logger.warning(f"[online] XGBoost update failed: {e} / XGBoost 更新失敗")
                return False
                
        elif 'lgb' in model_type:
            # LightGBM warm start / LightGBM 熱啟動
            try:
                model.set_params(n_estimators=model.n_estimators + 50)
                model.fit(X, y, init_model=model.booster_)
                logger.info(f"[online] {timeframe_label}: LightGBM updated with {len(X)} samples / LightGBM 已更新 {len(X)} 個樣本")
            except Exception as e:
                logger.warning(f"[online] LightGBM update failed: {e} / LightGBM 更新失敗")
                return False
        else:
            # For other models, just refit on recent data / 其他模型直接用近期數據重擬合
            try:
                model.fit(X, y)
                logger.info(f"[online] {timeframe_label}: Model refit with {len(X)} samples / 模型已重擬合 {len(X)} 個樣本")
            except Exception as e:
                logger.warning(f"[online] Refit failed: {e} / 重擬合失敗")
                return False
        
        # Save updated model / 儲存更新後的模型
        model_data['model'] = model
        model_data['online_updated'] = True
        model_data['online_update_date'] = datetime.now(HK_TZ).strftime('%Y-%m-%d')
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        # Also save online version / 同時儲存線上版本
        timestamp = datetime.now(HK_TZ).strftime('%Y%m%d_%H%M%S')
        online_path = os.path.join(ONLINE_DIR, f'model_{timeframe_label}_{timestamp}.pkl')
        with open(online_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"[online] {timeframe_label}: Update saved / 更新已儲存")
        return True
        
    except Exception as e:
        logger.error(f"[online] Update failed for {timeframe_label}: {e} / 更新失敗")
        return False


def compare_online_vs_full(stock_codes: list, timeframe_label: str, days: int) -> dict:
    """
    Compare online model performance vs full retrain.
    比較增量更新模型與全量重訓模型的表現。
    Returns comparison metrics / 返回比較指標。
    """
    from src.train_model import prepare_data
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import f1_score, roc_auc_score
    
    try:
        data, features = prepare_data(stock_codes, days)
        X = data[features]
        y = data['target']
        
        tscv = TimeSeriesSplit(n_splits=3)
        splits = list(tscv.split(X))
        train_idx, val_idx = splits[-1]
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Load current model (may be online-updated) / 載入當前模型 (可能已增量更新)
        model_path = os.path.join(MODELS_DIR, f'best_model_{timeframe_label}.pkl')
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        online_model = model_data['model']
        online_preds = online_model.predict(X_val)
        online_proba = online_model.predict_proba(X_val)[:, 1]
        online_f1 = f1_score(y_val, online_preds, zero_division=0)
        online_auc = roc_auc_score(y_val, online_proba) if len(np.unique(y_val)) > 1 else 0.5
        
        return {
            'online_f1': online_f1,
            'online_auc': online_auc,
            'model_type': model_data.get('model_type', 'unknown'),
        }
        
    except Exception as e:
        logger.warning(f"Comparison failed: {e} / 比較失敗")
        return {}
