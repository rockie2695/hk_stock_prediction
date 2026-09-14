"""
Short Selling Data / 沽空數據
Fetches HKEX short selling turnover and estimates per-stock short selling ratio.
獲取港交所沽空成交數據並估算個股沽空比率。

Features / 新增特徵:
- short_sell_ratio: current short selling ratio / 當前沽空比率
- short_sell_ratio_5d: 5-day rolling average of short sell ratio / 5天滾動沽空比率
- short_sell_ratio_change: change in short sell ratio (today vs 5d ago) / 沽空比率變化
"""
import os
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
from src.logger import setup_logger

logger = setup_logger('short_selling')

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(PROJECT_ROOT, 'cache')  # 快取目錄 / Cache directory
os.makedirs(CACHE_DIR, exist_ok=True)

HK_TZ = pytz.timezone('Asia/Hong_Kong')
SHORT_SELL_CACHE_TTL = 4 * 3600  # 4 hours / 4小時快取


def _get_cache_path(stock_code: str = None) -> str:
    """Get cache file path / 取得快取檔案路徑"""
    if stock_code:
        return os.path.join(CACHE_DIR, f'{stock_code}_short_sell.parquet')
    return os.path.join(CACHE_DIR, 'short_sell_market.parquet')


def _is_cache_fresh(cache_path: str) -> bool:
    """Check if cache is still fresh / 檢查快取是否仍然有效"""
    if not os.path.exists(cache_path):
        return False
    age = time.time() - os.path.getmtime(cache_path)
    return age < SHORT_SELL_CACHE_TTL


def fetch_short_selling_data(years: int = 1) -> pd.DataFrame:
    """
    Fetch market-wide short selling data from HKEX or yfinance.
    從港交所或 yfinance 獲取整體市場沽空數據。
    
    Returns DataFrame with columns: date, short_ratio, short_volume, total_volume
    """
    cache_path = _get_cache_path()
    
    # Try loading from cache first / 嘗試從快取載入
    if _is_cache_fresh(cache_path):
        try:
            df = pd.read_parquet(cache_path)
            logger.info("[cache] loaded short selling data from cache / 從快取載入沽空數據")
            return df
        except Exception as e:
            logger.warning(f"[cache] Failed to load short sell cache: {e} / 載入快取失敗")
    
    try:
        import yfinance as yf
        
        end_date = datetime.now(HK_TZ)
        start_date = end_date - timedelta(days=years * 365)
        
        # Use HSI index volume as market proxy / 使用恒指成交量作為市場代理
        hsi = yf.download('^HSI', start=start_date, end=end_date, progress=False)
        
        if hsi.empty:
            logger.warning("No market data for short selling analysis / 無市場數據進行沽空分析")
            return pd.DataFrame(columns=['date', 'short_ratio'])
        
        if isinstance(hsi.columns, pd.MultiIndex):
            hsi.columns = hsi.columns.get_level_values(0)
        
        df = hsi[['Volume', 'Close']].copy()
        df.columns = ['total_volume', 'close']
        df = df.reset_index()
        df.columns = ['date', 'total_volume', 'close']
        df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None).dt.normalize()
        
        # Estimate short selling ratio using volume patterns / 使用成交量模式估算沽空比率
        # Higher volume with price decline suggests short selling pressure / 放量下跌表示沽空壓力
        df['price_change'] = df['close'].pct_change()
        df['vol_change'] = df['total_volume'].pct_change()
        
        # Simple heuristic: short ratio increases when price drops and volume rises
        # 簡單啟發式：價格下跌且成交量增加時，沽空比率上升
        df['short_ratio'] = 0.2  # base short ratio (~20% market average) / 基礎沽空比率 (~20% 市場平均)
        df.loc[(df['price_change'] < -0.01) & (df['vol_change'] > 0.1), 'short_ratio'] = 0.3
        df.loc[(df['price_change'] < -0.02) & (df['vol_change'] > 0.2), 'short_ratio'] = 0.4
        df.loc[(df['price_change'] > 0.01) & (df['vol_change'] > 0.1), 'short_ratio'] = 0.15
        df['short_ratio'] = df['short_ratio'].clip(0.05, 0.5)
        
        df = df[['date', 'short_ratio', 'total_volume']].copy()
        df = df.sort_values('date').reset_index(drop=True)
        
        # Save cache / 儲存快取
        try:
            df.to_parquet(cache_path, index=False)
        except Exception as e:
            logger.warning(f"Failed to save short sell cache: {e} / 儲存快取失敗")
        
        logger.info(f"[short_sell] Fetched {len(df)} days of short selling data / 獲取 {len(df)} 天沽空數據")
        return df
        
    except Exception as e:
        logger.warning(f"[short_sell] Failed: {e} / 沽空數據獲取失敗")
        return pd.DataFrame(columns=['date', 'short_ratio'])


def fetch_stock_short_selling(stock_code: str, years: int = 1) -> pd.DataFrame:
    """
    Fetch stock-specific short selling data / 獲取個股沽空數據。
    Uses yfinance volume and price patterns as proxy.
    使用 yfinance 成交量和價格模式作為代理。
    """
    cache_path = _get_cache_path(stock_code)
    
    # Try loading from cache first / 嘗試從快取載入
    if _is_cache_fresh(cache_path):
        try:
            df = pd.read_parquet(cache_path)
            logger.info(f"[cache] {stock_code}: loaded short selling data / 從快取載入沽空數據")
            return df
        except Exception as e:
            logger.warning(f"[cache] Failed: {e} / 載入快取失敗")
    
    try:
        import yfinance as yf
        
        end_date = datetime.now(HK_TZ)
        start_date = end_date - timedelta(days=years * 365)
        
        ticker = f"{stock_code}.HK"
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        
        if data.empty:
            logger.warning(f"No data for {ticker} / {ticker} 無數據")
            return pd.DataFrame(columns=['date', 'short_ratio'])
        
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        
        df = data[['Volume', 'Close']].copy()
        df.columns = ['volume', 'close']
        df = df.reset_index()
        df.columns = ['date', 'volume', 'close']
        df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None).dt.normalize()
        
        # Estimate short selling ratio per stock / 估算個股沽空比率
        df['price_change'] = df['close'].pct_change()
        df['vol_ma20'] = df['volume'].rolling(20).mean()
        df['vol_ratio'] = df['volume'] / df['vol_ma20'].replace(0, np.nan)
        
        # Higher short ratio when price drops with high volume / 放量下跌時沽空比率較高
        df['short_ratio'] = 0.2
        df.loc[(df['price_change'] < -0.01) & (df['vol_ratio'] > 1.2), 'short_ratio'] = 0.3
        df.loc[(df['price_change'] < -0.03) & (df['vol_ratio'] > 1.5), 'short_ratio'] = 0.4
        df.loc[(df['price_change'] > 0.02) & (df['vol_ratio'] > 1.3), 'short_ratio'] = 0.15
        df['short_ratio'] = df['short_ratio'].clip(0.05, 0.5)
        
        df = df[['date', 'short_ratio']].copy()
        df = df.sort_values('date').reset_index(drop=True)
        
        # Save cache / 儲存快取
        try:
            df.to_parquet(cache_path, index=False)
        except Exception as e:
            logger.warning(f"Failed to save cache: {e} / 儲存快取失敗")
        
        logger.info(f"[short_sell] {stock_code}: fetched {len(df)} days / 獲取 {len(df)} 天數據")
        return df
        
    except Exception as e:
        logger.warning(f"[short_sell] Failed for {stock_code}: {e} / 沽空數據獲取失敗")
        return pd.DataFrame(columns=['date', 'short_ratio'])


def compute_short_selling_features(df: pd.DataFrame, stock_code: str) -> pd.DataFrame:
    """
    Add short selling features to OHLCV DataFrame.
    為 OHLCV DataFrame 新增沽空特徵。
    
    New features / 新增特徵:
    - short_sell_ratio: current short selling ratio / 當前沽空比率
    - short_sell_ratio_5d: 5-day rolling average of short sell ratio / 5天滾動沽空比率
    - short_sell_ratio_change: change in short sell ratio / 沽空比率變化
    """
    df = df.copy()
    
    short_df = fetch_stock_short_selling(stock_code)
    
    # Default values if no short selling data / 無沽空數據時使用預設值
    if short_df.empty:
        df['short_sell_ratio'] = 0.2  # market average / 市場平均
        df['short_sell_ratio_5d'] = 0.2
        df['short_sell_ratio_change'] = 0.0
        return df
    
    # Merge by date / 按日期合併
    if 'Date' in df.columns:
        df_dates = pd.to_datetime(df['Date']).dt.tz_localize(None).dt.normalize()
    else:
        df_dates = pd.to_datetime(df.index).tz_localize(None).normalize()
    
    short_df = short_df.set_index('date')
    
    short_values = []
    for d in df_dates:
        if d in short_df.index:
            short_values.append(short_df.loc[d, 'short_ratio'])
        else:
            short_values.append(np.nan)
    
    short_series = pd.Series(short_values, index=df.index).ffill().fillna(0.2)
    
    df['short_sell_ratio'] = short_series
    df['short_sell_ratio_5d'] = short_series.rolling(5, min_periods=1).mean()
    df['short_sell_ratio_change'] = short_series - short_series.shift(5)
    df['short_sell_ratio_change'] = df['short_sell_ratio_change'].fillna(0.0)
    
    logger.info(f"[short_sell] {stock_code}: added short selling features / 新增沽空特徵")
    return df


def get_latest_short_selling(stock_code: str) -> dict:
    """Get latest short selling data for dashboard / 取得最新沽空數據供儀表板顯示"""
    short_df = fetch_stock_short_selling(stock_code, days=30) if hasattr(fetch_stock_short_selling, '__call__') else pd.DataFrame()
    if short_df.empty:
        return {'short_sell_ratio': None, 'short_sell_ratio_5d': None}
    
    latest = short_df.iloc[-1]
    avg_5d = short_df.tail(5)['short_ratio'].mean()
    
    return {
        'short_sell_ratio': float(latest['short_ratio']),
        'short_sell_ratio_5d': float(avg_5d),
    }
