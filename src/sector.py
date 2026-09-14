"""
Sector Rotation Tracking / 板塊輪動追蹤
Tracks sector-level momentum for HK stocks using yfinance sector ETFs.
使用 yfinance 板塊 ETF 追蹤港股板塊動量。

Features / 新增特徵:
- sector_momentum_5d: 5-day return of stock's sector ETF / 股票所屬板塊ETF 5天報酬
- sector_momentum_20d: 20-day return of stock's sector ETF / 股票所屬板塊ETF 20天報酬
- sector_vs_hsi: sector return minus HSI return (relative strength) / 板塊相對恒指強弱
"""
import os
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
from src.logger import setup_logger

logger = setup_logger('sector')

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(PROJECT_ROOT, 'cache')  # 快取目錄 / Cache directory
os.makedirs(CACHE_DIR, exist_ok=True)

HK_TZ = pytz.timezone('Asia/Hong_Kong')
SECTOR_CACHE_TTL = 4 * 3600  # 4 hours / 4小時快取

# HK sector ETFs mapped to yfinance tickers / 港股板塊ETF對應yfinance代碼
SECTOR_ETFS = {
    'tech': '3033.HK',      # Hang Seng TECH Index ETF / 恒生科技指數ETF
    'finance': '3022.HK',   # Hang Seng China Enterprises ETF / 恒生中國企業ETF
    'property': '3048.HK',  # Hang Seng Property ETF / 恒生地產ETF
    'energy': '3046.HK',    # Hang Seng Energy ETF / 恒生能源ETF
    'healthcare': '3069.HK',# Hang Seng Healthcare ETF / 恒生醫療保健ETF
    'consumer': '3053.HK',  # Hang Seng Consumer ETF / 恒生消費ETF
}

# Stock to sector mapping (HK stocks) / 股票板塊對應表 (港股)
STOCK_SECTORS = {
    '0700': 'tech', '9988': 'tech', '3690': 'tech', '9618': 'tech',
    '1810': 'tech', '2382': 'tech', '0981': 'tech', '6060': 'tech',
    '0005': 'finance', '1398': 'finance', '3988': 'finance', '0939': 'finance',
    '1288': 'finance', '2318': 'finance', '0388': 'finance', '2628': 'finance',
    '0016': 'property', '0011': 'property', '0002': 'property', '0006': 'property',
    '0012': 'property', '0017': 'property', '0001': 'property',
    '0883': 'energy', '0857': 'energy', '2899': 'energy',
    '2269': 'healthcare', '1177': 'healthcare', '2359': 'healthcare',
    '0027': 'consumer', '2331': 'consumer', '6862': 'consumer',
}


def _get_cache_path() -> str:
    """Get cache file path / 取得快取檔案路徑"""
    return os.path.join(CACHE_DIR, 'sector_data.parquet')


def _is_cache_fresh() -> bool:
    """Check if cache is still fresh / 檢查快取是否仍然有效"""
    path = _get_cache_path()
    if not os.path.exists(path):
        return False
    age = time.time() - os.path.getmtime(path)
    return age < SECTOR_CACHE_TTL


def fetch_sector_data(years: int = 1) -> pd.DataFrame:
    """
    Fetch sector ETF price data / 獲取板塊ETF價格數據。
    Returns DataFrame with sector returns / 返回包含板塊報酬的DataFrame。
    """
    if _is_cache_fresh():
        try:
            df = pd.read_parquet(_get_cache_path())
            logger.info("[cache] loaded sector data from cache / 從快取載入板塊數據")
            return df
        except Exception as e:
            logger.warning(f"[cache] Failed to load sector cache: {e} / 載入快取失敗")
    
    try:
        import yfinance as yf
        
        end_date = datetime.now(HK_TZ)
        start_date = end_date - timedelta(days=years * 365)
        
        all_data = {}
        for sector_name, ticker in SECTOR_ETFS.items():
            try:
                data = yf.download(ticker, start=start_date, end=end_date, progress=False)
                if not data.empty:
                    if isinstance(data.columns, pd.MultiIndex):
                        data.columns = data.columns.get_level_values(0)
                    all_data[f'{sector_name}_close'] = data['Close']
                    logger.info(f"  {sector_name} ({ticker}): {len(data)} rows")
            except Exception as e:
                logger.warning(f"  {sector_name} ({ticker}) fetch failed: {e}")
        
        if not all_data:
            logger.warning("No sector data fetched / 未獲取板塊數據")
            return pd.DataFrame()
        
        sector_df = pd.DataFrame(all_data)
        sector_df = sector_df.ffill()
        
        if sector_df.index.tz is not None:
            sector_df.index = sector_df.index.tz_localize(None).normalize()
        
        # Compute returns / 計算報酬率
        for col in list(sector_df.columns):
            if col.endswith('_close'):
                sector_name = col.replace('_close', '')
                sector_df[f'{sector_name}_ret_5d'] = sector_df[col].pct_change(5)
                sector_df[f'{sector_name}_ret_20d'] = sector_df[col].pct_change(20)
        
        # Save cache / 儲存快取
        try:
            sector_df.to_parquet(_get_cache_path())
        except Exception as e:
            logger.warning(f"Failed to save sector cache: {e} / 儲存快取失敗")
        
        logger.info(f"[sector] Fetched data for {len(SECTOR_ETFS)} sectors / 獲取 {len(SECTOR_ETFS)} 個板塊數據")
        return sector_df
        
    except Exception as e:
        logger.warning(f"[sector] Failed to fetch sector data: {e} / 獲取板塊數據失敗")
        return pd.DataFrame()


def get_stock_sector(stock_code: str) -> str:
    """Get sector for a stock code / 取得股票代碼所屬板塊"""
    return STOCK_SECTORS.get(stock_code, 'tech')


def compute_sector_features(df: pd.DataFrame, stock_code: str) -> pd.DataFrame:
    """
    Add sector rotation features to OHLCV DataFrame.
    為 OHLCV DataFrame 新增板塊輪動特徵。
    
    New features / 新增特徵:
    - sector_momentum_5d: 5-day return of stock's sector ETF / 板塊ETF 5天報酬
    - sector_momentum_20d: 20-day return of stock's sector ETF / 板塊ETF 20天報酬
    - sector_vs_hsi: sector return minus HSI return (relative strength) / 板塊相對恒指強弱
    """
    df = df.copy()
    
    sector_data = fetch_sector_data()
    sector = get_stock_sector(stock_code)
    
    # Default values if no sector data / 無板塊數據時使用預設值
    if sector_data.empty or f'{sector}_ret_5d' not in sector_data.columns:
        df['sector_momentum_5d'] = 0.0
        df['sector_momentum_20d'] = 0.0
        df['sector_vs_hsi'] = 0.0
        return df
    
    # Map sector returns to stock dates / 將板塊報酬對應到股票日期
    if 'Date' in df.columns:
        dates = pd.to_datetime(df['Date']).dt.tz_localize(None).dt.normalize()
    else:
        dates = pd.to_datetime(df.index).tz_localize(None).normalize()
    
    sector_mom_5d = []
    sector_mom_20d = []
    sector_vs_hsi = []
    
    for d in dates:
        if d in sector_data.index:
            sector_mom_5d.append(sector_data.loc[d, f'{sector}_ret_5d'] if f'{sector}_ret_5d' in sector_data.columns else 0.0)
            sector_mom_20d.append(sector_data.loc[d, f'{sector}_ret_20d'] if f'{sector}_ret_20d' in sector_data.columns else 0.0)
            # Sector vs HSI (relative strength) / 板塊相對恒指強弱
            hsi_ret = sector_data.loc[d, 'hsi_ret_5d'] if 'hsi_ret_5d' in sector_data.columns else 0.0
            sector_ret = sector_data.loc[d, f'{sector}_ret_5d'] if f'{sector}_ret_5d' in sector_data.columns else 0.0
            sector_vs_hsi.append(sector_ret - hsi_ret)
        else:
            sector_mom_5d.append(0.0)
            sector_mom_20d.append(0.0)
            sector_vs_hsi.append(0.0)
    
    df['sector_momentum_5d'] = pd.Series(sector_mom_5d, index=df.index).fillna(0.0)
    df['sector_momentum_20d'] = pd.Series(sector_mom_20d, index=df.index).fillna(0.0)
    df['sector_vs_hsi'] = pd.Series(sector_vs_hsi, index=df.index).fillna(0.0)
    
    logger.info(f"[sector] {stock_code}: sector={sector}, added sector features / 板塊={sector}，新增板塊特徵")
    return df


def get_sector_summary(stock_code: str) -> dict:
    """Get latest sector info for dashboard display / 取得最新板塊資訊供儀表板顯示"""
    sector = get_stock_sector(stock_code)
    sector_data = fetch_sector_data()
    
    if sector_data.empty or f'{sector}_ret_5d' not in sector_data.columns:
        return {'sector': sector, 'sector_momentum_5d': None, 'sector_momentum_20d': None}
    
    latest = sector_data.iloc[-1]
    return {
        'sector': sector,
        'sector_momentum_5d': float(latest.get(f'{sector}_ret_5d', 0.0)),
        'sector_momentum_20d': float(latest.get(f'{sector}_ret_20d', 0.0)),
    }
