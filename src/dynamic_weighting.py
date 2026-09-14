"""
Dynamic Weighting - 動態集成權重
Adjusts ensemble model weights based on recent performance.
根據近期表現動態調整集成模型權重。

Tracks per-model accuracy and uses exponential moving average (EMA) for weighting.
追蹤每個模型的準確度，使用指數移動平均 (EMA) 進行權重調整。
Higher-performing models get higher weights in the ensemble.
表現較好的模型在集成中獲得較高權重。
"""
import os
import pickle
import json
import pandas as pd
import numpy as np
from datetime import datetime
import pytz
from src.logger import setup_logger

logger = setup_logger('dynamic_weighting')

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
WEIGHTS_FILE = os.path.join(MODELS_DIR, 'dynamic_weights.json')  # 動態權重檔案

HK_TZ = pytz.timezone('Asia/Hong_Kong')


def load_dynamic_weights() -> dict:
    """Load stored dynamic weights from file / 從檔案載入儲存的動態權重"""
    if not os.path.exists(WEIGHTS_FILE):
        return {}
    try:
        with open(WEIGHTS_FILE, 'r') as f:
            return json.load(f)
    except Exception:
        return {}


def save_dynamic_weights(weights: dict):
    """Save dynamic weights to file / 將動態權重儲存到檔案"""
    try:
        with open(WEIGHTS_FILE, 'w') as f:
            json.dump(weights, f, indent=2)
    except Exception as e:
        logger.warning(f"Failed to save weights: {e} / 儲存權重失敗")


def update_model_weights(stock_code: str, timeframe: str, model_accuracies: dict, alpha: float = 0.3) -> dict:
    """
    Update ensemble weights based on recent model performance.
    根據近期模型表現更新集成權重。
    
    Args / 參數:
        stock_code: Stock code / 股票代碼
        timeframe: '1d', '5d', or '20d' / 時間範圍
        model_accuracies: Dict mapping model names to recent accuracy scores
                         / 模型名稱到近期準確度分數的字典
        alpha: EMA decay factor (higher = more weight on recent performance)
              / EMA 衰減因子 (越高 = 近期表現權重越大)
        
    Returns / 返回:
        Updated weights dict / 更新後的權重字典
    """
    key = f"{stock_code}_{timeframe}"
    stored = load_dynamic_weights()
    
    if key in stored:
        old_weights = stored[key]['weights']
    else:
        # Equal weights as default / 預設使用等權重
        n_models = len(model_accuracies)
        old_weights = {name: 1.0 / n_models for name in model_accuracies}
    
    # EMA update / EMA 更新
    new_weights = {}
    for name, acc in model_accuracies.items():
        old_w = old_weights.get(name, 1.0 / len(model_accuracies))
        new_w = alpha * acc + (1 - alpha) * old_w
        new_weights[name] = max(new_w, 0.1)  # minimum weight floor / 最低權重下限
    
    # Normalize to sum to 1 / 正規化使總和為1
    total = sum(new_weights.values())
    new_weights = {name: w / total for name, w in new_weights.items()}
    
    stored[key] = {
        'weights': new_weights,
        'accuracies': model_accuracies,
        'updated': datetime.now(HK_TZ).isoformat(),
    }
    
    save_dynamic_weights(stored)
    logger.info(f"[dynamic] Updated weights for {key}: {new_weights} / 已更新權重")
    
    return new_weights


def get_dynamic_weights(stock_code: str, timeframe: str) -> dict:
    """Get current dynamic weights for a stock/timeframe pair / 取得股票/時間範圍的當前動態權重"""
    key = f"{stock_code}_{timeframe}"
    stored = load_dynamic_weights()
    
    if key in stored:
        return stored[key]['weights']
    
    # Default equal weights / 預設等權重
    return {'xgb': 0.25, 'lgb': 0.25, 'rf': 0.25, 'cb': 0.25}


def compute_dynamic_ensemble_prediction(model_data: dict, X: pd.DataFrame, stock_code: str, timeframe: str) -> np.ndarray:
    """
    Make prediction using dynamically weighted ensemble.
    使用動態加權集成進行預測。
    
    Args / 參數:
        model_data: Current model data dict / 當前模型數據字典
        X: Feature DataFrame / 特徵 DataFrame
        stock_code: Stock code / 股票代碼
        timeframe: Timeframe label / 時間範圍標籤
        
    Returns / 返回:
        Weighted probability array / 加權機率陣列
    """
    model = model_data.get('model')
    
    if not hasattr(model, 'estimators_'):
        # Not an ensemble, return direct prediction / 非集成模型，直接返回預測
        return model.predict_proba(X)[:, 1]
    
    # Get individual model predictions / 取得各模型預測
    weights = get_dynamic_weights(stock_code, timeframe)
    
    weighted_proba = np.zeros(len(X))
    total_weight = 0
    
    for name, estimator in model.estimators_:
        try:
            proba = estimator.predict_proba(X)[:, 1]
            weight = weights.get(name, 1.0 / len(model.estimators_))
            weighted_proba += proba * weight
            total_weight += weight
        except Exception as e:
            logger.warning(f"Failed to get prediction from {name}: {e} / 無法取得 {name} 預測")
            continue
    
    if total_weight > 0:
        weighted_proba /= total_weight
    
    return weighted_proba


def evaluate_model_performance(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Calculate model performance metrics for weight updates / 計算模型表現指標用於權重更新"""
    from sklearn.metrics import f1_score, accuracy_score
    
    return {
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'accuracy': accuracy_score(y_true, y_pred),
    }
