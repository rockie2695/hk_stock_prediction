"""
Data fetcher - downloads Hong Kong stock historical daily data.
Primary: akshare | Fallback: yfinance
Includes retry mechanism.
"""
import time
import pandas as pd
import pytz
from datetime import datetime, timedelta
from src.logger import setup_logger

logger = setup_logger('data_fetcher')
HK_TZ = pytz.timezone('Asia/Hong_Kong')


def fetch_stock_data(stock_code: str, years: int = 3) -> pd.DataFrame:
    """
    Fetch historical daily OHLCV data for a Hong Kong stock.

    Args:
        stock_code: Stock code like '0700' (without exchange prefix)
        years: Number of years of historical data to fetch

    Returns:
        DataFrame with columns: Date, Open, High, Low, Close, Volume (sorted ascending)
    """
    end_date = datetime.now(HK_TZ)
    start_date = end_date - timedelta(days=years * 365)

    # Try akshare first, then yfinance
    df = None
    for attempt in range(1, 4):
        try:
            df = _fetch_akshare(stock_code, start_date, end_date)
            if df is not None and len(df) > 0:
                logger.info(f"[akshare] {stock_code}: fetched {len(df)} rows")
                return df
        except Exception as e:
            logger.warning(f"[akshare] Attempt {attempt}/3 failed for {stock_code}: {e}")
            if attempt < 3:
                time.sleep(5)

    # Fallback to yfinance
    for attempt in range(1, 4):
        try:
            df = _fetch_yfinance(stock_code, start_date, end_date)
            if df is not None and len(df) > 0:
                logger.info(f"[yfinance] {stock_code}: fetched {len(df)} rows")
                return df
        except Exception as e:
            logger.warning(f"[yfinance] Attempt {attempt}/3 failed for {stock_code}: {e}")
            if attempt < 3:
                time.sleep(5)

    raise RuntimeError(
        f"Failed to fetch data for {stock_code} after 3 attempts with both akshare and yfinance."
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

    ticker = f"{int(stock_code)}.HK"
    data = yf.download(ticker, start=start_date.strftime('%Y-%m-%d'),
                       end=end_date.strftime('%Y-%m-%d'), progress=False)

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
