"""
Feature engineering - computes technical indicators from OHLCV data.
CRITICAL: No look-ahead bias. All features use only past/current data.
"""
import pandas as pd
import numpy as np
from src.logger import setup_logger

logger = setup_logger('feature_engineering')

FEATURE_COLUMNS = [
    'ret_5d', 'ret_10d', 'ret_20d',
    'vol_ratio_5d',
    'rsi_14',
    'macd_diff', 'macd_dea', 'macd_hist',
    'bb_width',
    'atr_14'
]


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute technical features from OHLCV data.
    All features use only historical data (no future data leakage).

    Args:
        df: DataFrame with Date, Open, High, Low, Close, Volume columns

    Returns:
        DataFrame with original columns plus feature columns
    """
    df = df.copy()

    # --- Return features ---
    df['ret_5d'] = df['Close'].pct_change(5)
    df['ret_10d'] = df['Close'].pct_change(10)
    df['ret_20d'] = df['Close'].pct_change(20)

    # --- Volume ratio (current / 5-day average) ---
    df['vol_ratio_5d'] = df['Volume'] / df['Volume'].rolling(5).mean()

    # --- RSI (14-day) using Wilder's smoothing ---
    df['rsi_14'] = _compute_rsi(df['Close'], period=14)

    # --- MACD (12, 26, 9) ---
    macd_diff, macd_dea, macd_hist = _compute_macd(df['Close'])
    df['macd_diff'] = macd_diff
    df['macd_dea'] = macd_dea
    df['macd_hist'] = macd_hist

    # --- Bollinger Band Width ---
    sma20 = df['Close'].rolling(20).mean()
    std20 = df['Close'].rolling(20).std()
    upper = sma20 + 2 * std20
    lower = sma20 - 2 * std20
    df['bb_width'] = (upper - lower) / sma20

    # --- ATR (14-day) ---
    df['atr_14'] = _compute_atr(df, period=14)

    logger.info(f"Computed {len(FEATURE_COLUMNS)} features, shape: {df.shape}")
    return df


def compute_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute binary target: 1 if next day's close > today's close, else 0.
    """
    df = df.copy()
    df['target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    return df


def _compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Compute RSI using Wilder's smoothing method."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)

    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def _compute_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """Compute MACD line, signal line, and histogram."""
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_diff = ema_fast - ema_slow
    macd_dea = macd_diff.ewm(span=signal, adjust=False).mean()
    macd_hist = macd_diff - macd_dea
    return macd_diff, macd_dea, macd_hist


def _compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Compute Average True Range."""
    high = df['High']
    low = df['Low']
    prev_close = df['Close'].shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = tr.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    return atr
