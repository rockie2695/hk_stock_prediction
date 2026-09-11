"""
Data fetcher - downloads Hong Kong stock historical daily data.
Primary: yfinance | Fallback: akshare
Includes retry mechanism and local parquet caching.
"""
import os
import time
import pandas as pd
import pytz
from datetime import datetime, timedelta
from src.logger import setup_logger

logger = setup_logger('data_fetcher')
HK_TZ = pytz.timezone('Asia/Hong_Kong')

# Cache directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(PROJECT_ROOT, 'cache')
os.makedirs(CACHE_DIR, exist_ok=True)

# Cache freshness: 4 hours during trading days, 24 hours otherwise
TRADING_HOURS_CACHE_TTL = 4 * 3600  # 4 hours
NON_TRADING_CACHE_TTL = 24 * 3600   # 24 hours


def _get_cache_path(stock_code: str) -> str:
    """Get cache file path for a stock."""
    return os.path.join(CACHE_DIR, f'{stock_code}_ohlcv.parquet')


def _is_cache_fresh(cache_path: str) -> bool:
    """Check if cache file is fresh enough to use."""
    if not os.path.exists(cache_path):
        return False

    mtime = os.path.getmtime(cache_path)
    age = time.time() - mtime

    # During trading hours (Mon-Fri 9:30-16:00 HKT), use shorter TTL
    now = datetime.now(HK_TZ)
    is_trading_day = now.weekday() < 5
    is_trading_hours = is_trading_day and 9 <= now.hour < 16

    ttl = TRADING_HOURS_CACHE_TTL if is_trading_hours else NON_TRADING_CACHE_TTL
    return age < ttl


def fetch_stock_data(stock_code: str, years: int = 3, use_cache: bool = True) -> pd.DataFrame:
    """
    Fetch historical daily OHLCV data for a Hong Kong stock.

    Args:
        stock_code: Stock code like '0700' (without exchange prefix)
        years: Number of years of historical data to fetch
        use_cache: Whether to use local parquet cache

    Returns:
        DataFrame with columns: Date, Open, High, Low, Close, Volume (sorted ascending)
    """
    cache_path = _get_cache_path(stock_code)

    # Try cache first
    if use_cache and _is_cache_fresh(cache_path):
        try:
            df = pd.read_parquet(cache_path)
            df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
            logger.info(f"[cache] {stock_code}: loaded {len(df)} rows from cache")
            return df
        except Exception as e:
            logger.warning(f"[cache] Failed to load cache for {stock_code}: {e}")

    end_date = datetime.now(HK_TZ)
    start_date = end_date - timedelta(days=years * 365)

    # Try yfinance first, then akshare
    df = None
    for attempt in range(1, 4):
        try:
            df = _fetch_yfinance(stock_code, start_date, end_date)
            if df is not None and len(df) > 0:
                logger.info(f"[yfinance] {stock_code}: fetched {len(df)} rows")
                # Save to cache
                if use_cache:
                    try:
                        df.to_parquet(cache_path, index=False)
                        logger.info(f"[cache] {stock_code}: saved to cache")
                    except Exception as e:
                        logger.warning(f"[cache] Failed to save cache for {stock_code}: {e}")
                return df
        except Exception as e:
            logger.warning(f"[yfinance] Attempt {attempt}/3 failed for {stock_code}: {e}")
            if attempt < 3:
                time.sleep(5)

    # Fallback to akshare
    for attempt in range(1, 4):
        try:
            df = _fetch_akshare(stock_code, start_date, end_date)
            if df is not None and len(df) > 0:
                logger.info(f"[akshare] {stock_code}: fetched {len(df)} rows")
                # Save to cache
                if use_cache:
                    try:
                        df.to_parquet(cache_path, index=False)
                        logger.info(f"[cache] {stock_code}: saved to cache")
                    except Exception as e:
                        logger.warning(f"[cache] Failed to save cache for {stock_code}: {e}")
                return df
        except Exception as e:
            logger.warning(f"[akshare] Attempt {attempt}/3 failed for {stock_code}: {e}")
            if attempt < 3:
                time.sleep(5)

    raise RuntimeError(
        f"Failed to fetch data for {stock_code} after 3 attempts with both yfinance and akshare."
    )


def _fetch_akshare(stock_code: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """Fetch using akshare."""
    import akshare as ak

    symbol = str(int(stock_code))  # Strip leading zeros
    start_str = start_date.strftime('%Y%m%d')
    end_str = end_date.strftime('%Y%m%d')

    df = ak.stock_hk_hist(
        symbol=symbol,
        period="daily",
        start_date=start_str,
        end_date=end_str,
        adjust="qfq"
    )

    # Standardize columns
    col_map = {}
    for col in df.columns:
        cl = col.lower()
        if '日期' in cl or 'date' in cl:
            col_map[col] = 'Date'
        elif '开盘' in cl or 'open' in cl:
            col_map[col] = 'Open'
        elif '最高' in cl or 'high' in cl:
            col_map[col] = 'High'
        elif '最低' in cl or 'low' in cl:
            col_map[col] = 'Low'
        elif '收盘' in cl or 'close' in cl:
            col_map[col] = 'Close'
        elif '成交量' in cl or 'volume' in cl:
            col_map[col] = 'Volume'

    df = df.rename(columns=col_map)

    required = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    return df[required]


def _fetch_yfinance(stock_code: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """Fetch using yfinance as fallback."""
    import yfinance as yf

    # HK stocks on Yahoo Finance use 4-digit codes: 0700.HK, 9988.HK, etc.
    ticker = f"{stock_code}.HK"
    data = yf.download(ticker, start=start_date.strftime('%Y-%m-%d'),
                       end=end_date.strftime('%Y-%m-%d'), progress=False,
                       auto_adjust=False)

    if data.empty:
        raise RuntimeError(f"No data returned from yfinance for {ticker}")

    # Flatten MultiIndex columns if present
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data = data.reset_index()
    data = data.rename(columns={
        'Date': 'Date',
        'Open': 'Open',
        'High': 'High',
        'Low': 'Low',
        'Close': 'Close',
        'Volume': 'Volume'
    })

    data['Date'] = pd.to_datetime(data['Date'])
    data = data.sort_values('Date').reset_index(drop=True)
    return data[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
