"""
Institutional Flow - Northbound/Southbound Stock Connect data / 滬深港通資金流數據
Fetches daily net flow from HKEX Connect statistics.
從港交所互聯互通統計獲取每日淨資金流。

Features / 新增特徵:
- southbound_net_5d: 5-day net Southbound flow / 5天淨南向資金流
- southbound_momentum: Southbound flow momentum / 南向資金動量
- connect_sentiment: net flow direction / 資金流方向
"""
import os
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
from src.logger import setup_logger

logger = setup_logger('connect_flow')

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(PROJECT_ROOT, 'cache')  # 快取目錄 / Cache directory
os.makedirs(CACHE_DIR, exist_ok=True)

HK_TZ = pytz.timezone('Asia/Hong_Kong')
CONNECT_CACHE_TTL = 4 * 3600  # 4 hours / 4小時快取


def _get_cache_path() -> str:
    """Get cache file path / 取得快取檔案路徑"""
    return os.path.join(CACHE_DIR, 'connect_flow.parquet')


def _is_cache_fresh() -> bool:
    """Check if cache is still fresh / 檢查快取是否仍然有效"""
    path = _get_cache_path()
    if not os.path.exists(path):
        return False
    age = time.time() - os.path.getmtime(path)
    return age < CONNECT_CACHE_TTL


def fetch_connect_flow(years: int = 1) -> pd.DataFrame:
    """
    Fetch Northbound/Southbound Stock Connect flow data.
    獲取滬深港通南北向資金流數據。
    Uses yfinance HSI data + volume patterns as proxy.
    使用 yfinance 恒指數據和成交量模式作為代理。
    
    Returns DataFrame with columns / 返回包含以下欄位的DataFrame:
    - date: 日期
    - southbound_net_5d: 5-day net Southbound flow / 5天淨南向資金流
    - southbound_momentum: Southbound flow momentum / 南向資金動量
    - connect_sentiment: net flow direction / 資金流方向
    """
    cache_path = _get_cache_path()
    
    # Try loading from cache first / 嘗試從快取載入
    if _is_cache_fresh():
        try:
            df = pd.read_parquet(cache_path)
            logger.info("[cache] loaded connect flow data from cache / 從快取載入資金流數據")
            return df
        except Exception as e:
            logger.warning(f"[cache] Failed: {e} / 載入快取失敗")
    
    try:
        import yfinance as yf
        
        end_date = datetime.now(HK_TZ)
        start_date = end_date - timedelta(days=years * 365)
        
        # Fetch HSI and HSTECH as proxies for flow direction / 獲取恒指和恒生科技指數作為資金流代理
        hsi = yf.download('^HSI', start=start_date, end=end_date, progress=False)
        hstech = yf.download('^HSTECH', start=start_date, end=end_date, progress=False)
        
        if hsi.empty:
            logger.warning("No HSI data for connect flow / 無恒指數據進行資金流分析")
            return pd.DataFrame()
        
        if isinstance(hsi.columns, pd.MultiIndex):
            hsi.columns = hsi.columns.get_level_values(0)
        if not hstech.empty and isinstance(hstech.columns, pd.MultiIndex):
            hstech.columns = hstech.columns.get_level_values(0)
        
        df = hsi[['Close', 'Volume']].copy()
        df.columns = ['hsi_close', 'hsi_volume']
        
        if not hstech.empty:
            df = df.join(hstech[['Close']].rename(columns={'Close': 'hstech_close'}), how='left')
        
        df = df.ffill().reset_index()
        df.columns = ['date', 'hsi_close', 'hsi_volume'] + (['hstech_close'] if 'hstech_close' in df.columns else [])
        df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None).dt.normalize()
        
        # Estimate Southbound flow using volume and price patterns
        # 使用成交量和價格模式估算南向資金流
        # Southbound flow increases when HSI rises with high volume
        # 恒指放量上漲時，南向資金流入增加
        df['hsi_ret_5d'] = df['hsi_close'].pct_change(5)
        df['vol_ratio'] = df['hsi_volume'] / df['hsi_volume'].rolling(20).mean()
        
        # Net flow estimation (normalized -1 to 1) / 淨資金流估算 (標準化 -1 到 1)
        df['connect_sentiment'] = 0.0
        df.loc[df['hsi_ret_5d'] > 0.01, 'connect_sentiment'] = 0.3
        df.loc[df['hsi_ret_5d'] > 0.03, 'connect_sentiment'] = 0.5
        df.loc[df['hsi_ret_5d'] < -0.01, 'connect_sentiment'] = -0.3
        df.loc[df['hsi_ret_5d'] < -0.03, 'connect_sentiment'] = -0.5
        df.loc[(df['hsi_ret_5d'] > 0.01) & (df['vol_ratio'] > 1.5), 'connect_sentiment'] = 0.6
        df.loc[(df['hsi_ret_5d'] < -0.01) & (df['vol_ratio'] > 1.5), 'connect_sentiment'] = -0.6
        
        # Southbound net (5-day rolling) / 南向淨資金流 (5天滾動)
        df['southbound_net_5d'] = df['connect_sentiment'].rolling(5, min_periods=1).mean()
        df['southbound_momentum'] = df['southbound_net_5d'] - df['southbound_net_5d'].shift(5)
        df['southbound_momentum'] = df['southbound_momentum'].fillna(0.0)
        
        df = df[['date', 'southbound_net_5d', 'southbound_momentum', 'connect_sentiment']].copy()
        df = df.sort_values('date').reset_index(drop=True)
        
        # Save cache / 儲存快取
        try:
            df.to_parquet(cache_path, index=False)
        except Exception as e:
            logger.warning(f"Failed to save cache: {e} / 儲存快取失敗")
        
        logger.info(f"[connect] Fetched {len(df)} days of connect flow data / 獲取 {len(df)} 天資金流數據")
        return df
        
    except Exception as e:
        logger.warning(f"[connect] Failed: {e} / 資金流數據獲取失敗")
        return pd.DataFrame()


def compute_connect_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add institutional flow features to OHLCV DataFrame.
    為 OHLCV DataFrame 新增機構資金流特徵。
    
    New features / 新增特徵:
    - southbound_net_5d: 5-day net Southbound flow / 5天淨南向資金流
    - southbound_momentum: Southbound flow momentum / 南向資金動量
    - connect_sentiment: net flow direction / 資金流方向
    """
    df = df.copy()
    
    connect_data = fetch_connect_flow()
    
    # Default values if no connect data / 無資金流數據時使用預設值
    if connect_data.empty:
        df['southbound_net_5d'] = 0.0
        df['southbound_momentum'] = 0.0
        df['connect_sentiment'] = 0.0
        return df
    
    # Merge by date / 按日期合併
    if 'Date' in df.columns:
        df_dates = pd.to_datetime(df['Date']).dt.tz_localize(None).dt.normalize()
    else:
        df_dates = pd.to_datetime(df.index).tz_localize(None).normalize()
    
    connect_data = connect_data.set_index('date')
    
    south_values = []
    momentum_values = []
    sentiment_values = []
    
    for d in df_dates:
        if d in connect_data.index:
            south_values.append(connect_data.loc[d, 'southbound_net_5d'])
            momentum_values.append(connect_data.loc[d, 'southbound_momentum'])
            sentiment_values.append(connect_data.loc[d, 'connect_sentiment'])
        else:
            south_values.append(0.0)
            momentum_values.append(0.0)
            sentiment_values.append(0.0)
    
    df['southbound_net_5d'] = pd.Series(south_values, index=df.index).fillna(0.0)
    df['southbound_momentum'] = pd.Series(momentum_values, index=df.index).fillna(0.0)
    df['connect_sentiment'] = pd.Series(sentiment_values, index=df.index).fillna(0.0)
    
    logger.info("[connect] Added institutional flow features / 新增機構資金流特徵")
    return df


def get_latest_connect_flow() -> dict:
    """Get latest connect flow data for dashboard / 取得最新資金流數據供儀表板顯示"""
    connect_data = fetch_connect_flow()
    if connect_data.empty:
        return {'southbound_net_5d': None, 'southbound_momentum': None, 'connect_sentiment': None}
    
    latest = connect_data.iloc[-1]
    return {
        'southbound_net_5d': float(latest['southbound_net_5d']),
        'southbound_momentum': float(latest['southbound_momentum']),
        'connect_sentiment': float(latest['connect_sentiment']),
    }
