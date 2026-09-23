"""
Market Regime Detection / 市場狀態偵測
Classifies market into bull/bear/sideways using HSI index data.
使用恒指數據將市場分類為牛市/熊市/震盪。

Uses MA crossover and volatility regime analysis.
使用均線交叉和波動率狀態分析。

Features / 新增特徵:
- market_regime: 0=bear, 1=sideways, 2=bull / 0=熊市, 1=震盪, 2=牛市
- regime_confidence: 0-1 confidence score / 信心度分數
- hsi_trend_50_200: 50-day MA / 200-day MA ratio - 1 / 50日均線/200日均線比率
"""
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
from src.logger import setup_logger

logger = setup_logger('regime')

HK_TZ = pytz.timezone('Asia/Hong_Kong')


def detect_regime(hsi_close: pd.Series) -> pd.DataFrame:
    """
    Detect market regime from HSI close prices.
    從恒指收盤價偵測市場狀態。
    
    Methods / 方法:
    1. MA Crossover: 50-day MA vs 200-day MA / 均線交叉：50日 vs 200日均線
    2. Volatility regime: low/medium/high / 波動率狀態：低/中/高
    3. Trend strength: ADX-based / 趨勢強度：基於 ADX
    
    Returns DataFrame with / 返回包含以下欄位的DataFrame:
    - market_regime: 0=bear(熊市), 1=sideways(震盪), 2=bull(牛市)
    - regime_confidence: 0-1 confidence score / 信心度分數
    - hsi_trend_50_200: 50-day MA / 200-day MA ratio - 1 / 均線比率
    """
    df = pd.DataFrame(index=hsi_close.index)
    
    # 50-day and 200-day MA / 50日和200日均線
    ma50 = hsi_close.rolling(50).mean()
    ma200 = hsi_close.rolling(200).mean()
    
    # Trend ratio / 趨勢比率
    df['hsi_trend_50_200'] = (ma50 / ma200.replace(0, np.nan)) - 1
    
    # MA crossover signal / 均線交叉信號
    ma_signal = (ma50 > ma200).astype(float)
    
    # Price position relative to MAs / 價格相對於均線的位置
    price_above_ma50 = (hsi_close > ma50).astype(float)
    price_above_ma200 = (hsi_close > ma200).astype(float)
    
    # Volatility regime (20-day rolling std of returns) / 波動率狀態 (20天滾動標準差)
    returns = hsi_close.pct_change(fill_method=None)
    volatility = returns.rolling(20).std()
    vol_percentile = volatility.rolling(60).rank(pct=True)
    
    # Regime classification / 狀態分類
    # Bull: MA50 > MA200, price above both MAs, rising momentum / 牛市：MA50>MA200，價格在均線上，上升動能
    # Bear: MA50 < MA200, price below both MAs, falling momentum / 熊市：MA50<MA200，價格在均線下，下降動能
    # Sideways: mixed signals / 震盪：信號混合
    regime_score = pd.Series(1.0, index=hsi_close.index)  # start sideways / 預設震盪
    
    # Bullish signals / 看多信號
    regime_score = regime_score + ma_signal * 0.3
    regime_score = regime_score + price_above_ma50 * 0.2
    regime_score = regime_score + price_above_ma200 * 0.2
    
    # Trend momentum (5-day return) / 趨勢動量 (5天報酬)
    momentum = hsi_close.pct_change(5, fill_method=None)
    regime_score = regime_score + (momentum > 0.01).astype(float) * 0.15
    regime_score = regime_score - (momentum < -0.01).astype(float) * 0.15
    
    # Map to regime / 映射到狀態
    df['market_regime'] = 1  # default sideways / 預設震盪
    df.loc[regime_score > 1.5, 'market_regime'] = 2  # bull / 牛市
    df.loc[regime_score < 0.5, 'market_regime'] = 0  # bear / 熊市
    
    # Confidence based on regime score distance from 1.0 / 基於分數距離的信心度
    df['regime_confidence'] = 1.0 - np.abs(regime_score - 1.0) / 1.5
    df['regime_confidence'] = df['regime_confidence'].clip(0.0, 1.0)
    
    return df


def compute_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add regime detection features to DataFrame.
    為 DataFrame 新增市場狀態偵測特徵。
    Requires HSI close prices (may come from market data merge).
    需要恒指收盤價 (可來自市場數據合併)。
    
    New features / 新增特徵:
    - market_regime: 0=bear(熊市), 1=sideways(震盪), 2=bull(牛市)
    - regime_confidence: confidence of regime classification / 狀態分類信心度
    - hsi_trend_50_200: HSI 50-day MA vs 200-day MA / 恒指50日 vs 200日均線
    """
    df = df.copy()
    
    # Try to use HSI data if available / 嘗試使用恒指數據
    if 'hsi_ret_5d' in df.columns:
        # Reconstruct approximate HSI from returns / 從報酬率重建近似恒指
        hsi_proxy = (1 + df['hsi_ret_5d'].fillna(0)).cumprod() * 1000
        regime_df = detect_regime(hsi_proxy)
        
        df['market_regime'] = regime_df['market_regime'].values
        df['regime_confidence'] = regime_df['regime_confidence'].values
        df['hsi_trend_50_200'] = regime_df['hsi_trend_50_200'].values
    else:
        # Default values when HSI data not available / 無恒指數據時使用預設值
        df['market_regime'] = 1.0
        df['regime_confidence'] = 0.5
        df['hsi_trend_50_200'] = 0.0
    
    # Fill NaN / 填補NaN
    df['market_regime'] = df['market_regime'].fillna(1.0)
    df['regime_confidence'] = df['regime_confidence'].fillna(0.5)
    df['hsi_trend_50_200'] = df['hsi_trend_50_200'].fillna(0.0)
    
    return df


def get_current_regime() -> dict:
    """Get current market regime for dashboard display / 取得當前市場狀態供儀表板顯示"""
    try:
        import yfinance as yf
        
        end_date = datetime.now(HK_TZ)
        start_date = end_date - timedelta(days=365 * 2)
        
        hsi = yf.download('^HSI', start=start_date, end=end_date, progress=False, auto_adjust=False)
        
        if hsi.empty:
            return {'regime': 'sideways', 'confidence': 0.5, 'trend': 0.0}
        
        if isinstance(hsi.columns, pd.MultiIndex):
            hsi.columns = hsi.columns.get_level_values(0)

        # yfinance can append incomplete rows (NaN Close); rolling MAs over
        # those produce NaN trend — drop them before detection.
        close = hsi['Close'].dropna()
        if close.empty:
            return {'regime': 'sideways', 'confidence': 0.5, 'trend': 0.0}

        regime_df = detect_regime(close)
        latest = regime_df.iloc[-1]

        regime_names = {0: 'bear', 1: 'sideways', 2: 'bull'}
        trend = float(latest['hsi_trend_50_200'])
        if not np.isfinite(trend):
            # e.g. <200 rows of history so MA200 never forms
            trend = 0.0
        confidence = float(latest['regime_confidence'])
        if not np.isfinite(confidence):
            confidence = 0.5
        return {
            'regime': regime_names.get(int(latest['market_regime']), 'sideways'),
            'confidence': confidence,
            'trend': trend,
        }
    except Exception as e:
        logger.warning(f"Failed to get regime: {e} / 無法取得市場狀態")
        return {'regime': 'sideways', 'confidence': 0.5, 'trend': 0.0}
