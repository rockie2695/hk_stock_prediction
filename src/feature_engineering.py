"""
Feature engineering - computes technical indicators from OHLCV data.
CRITICAL: No look-ahead bias. All features use only past/current data.
"""
import pandas as pd
import numpy as np
from src.logger import setup_logger

logger = setup_logger('feature_engineering')

# All feature columns
FEATURE_COLUMNS = [
    # Price features
    'ret_1d', 'ret_3d', 'ret_5d', 'ret_10d', 'ret_20d', 'ret_30d',
    'high_low_range', 'close_to_high', 'close_to_low',
    # Volume features
    'vol_ratio_5d', 'vol_ratio_10d',
    'obv_change',
    'volume_cv',
    # RSI
    'rsi_14',
    # MACD
    'macd_diff', 'macd_dea', 'macd_hist',
    # Bollinger
    'bb_width',
    # ATR
    'atr_14', 'atr_ratio',
    # Stochastic
    'stoch_k', 'stoch_d',
    # ADX
    'adx',
    # MFI
    'mfi',
    # Williams %R
    'williams_r',
    # Price position
    'ma50_deviation',
    # Rolling statistics
    'ret_5d_skew', 'ret_5d_kurt',
    'volatility_10d', 'volatility_20d',
    # Market context (added dynamically if available)
    # 'hsi_ret_5d', 'usdhkd_change',
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

    # --- Price features ---
    df['ret_1d'] = df['Close'].pct_change(1)
    df['ret_3d'] = df['Close'].pct_change(3)
    df['ret_5d'] = df['Close'].pct_change(5)
    df['ret_10d'] = df['Close'].pct_change(10)
    df['ret_20d'] = df['Close'].pct_change(20)
    df['ret_30d'] = df['Close'].pct_change(30)

    # Intraday range
    df['high_low_range'] = (df['High'] - df['Low']) / df['Close']
    df['close_to_high'] = (df['High'] - df['Close']) / df['Close']
    df['close_to_low'] = (df['Close'] - df['Low']) / df['Close']

    # --- Volume features ---
    df['vol_ratio_5d'] = df['Volume'] / df['Volume'].rolling(5).mean()
    df['vol_ratio_10d'] = df['Volume'] / df['Volume'].rolling(10).mean()

    # Volume Coefficient of Variation (CV)
    df['volume_cv'] = df['Volume'].rolling(20).std() / df['Volume'].rolling(20).mean()

    # OBV change
    obv = _compute_obv(df)
    df['obv_change'] = obv.pct_change(5)

    # --- RSI (14-day) ---
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

    # --- ATR Ratio (ATR / Close Price) ---
    df['atr_ratio'] = df['atr_14'] / df['Close']

    # --- Stochastic (14, 3, 3) ---
    stoch_k, stoch_d = _compute_stochastic(df, k_period=14, d_period=3)
    df['stoch_k'] = stoch_k
    df['stoch_d'] = stoch_d

    # --- ADX (14) ---
    df['adx'] = _compute_adx(df, period=14)

    # --- MFI (14) ---
    df['mfi'] = _compute_mfi(df, period=14)

    # --- Williams %R (14) ---
    df['williams_r'] = _compute_williams_r(df, period=14)

    # --- Price Position: Deviation from 50-day MA ---
    df['ma50_deviation'] = (df['Close'] - df['Close'].rolling(50).mean()) / df['Close'].rolling(50).mean()

    # --- Rolling statistics ---
    df['ret_5d_skew'] = df['ret_1d'].rolling(5).skew()
    df['ret_5d_kurt'] = df['ret_1d'].rolling(5).kurt()
    df['volatility_10d'] = df['ret_1d'].rolling(10).std()
    df['volatility_20d'] = df['ret_1d'].rolling(20).std()

    logger.info(f"Computed {len(FEATURE_COLUMNS)} features, shape: {df.shape}")
    return df


def compute_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute binary target: 1 if next day's close > today's close, else 0.
    """
    df = df.copy()
    df['target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    return df


def compute_target_days(df: pd.DataFrame, days: int) -> pd.DataFrame:
    """
    Compute binary target for N days ahead.
    1 if price N days later > today's close, else 0.

    Args:
        df: DataFrame with Close column
        days: Number of days ahead to predict (1, 5, or 20)

    Returns:
        DataFrame with 'target' column
    """
    df = df.copy()
    df['target'] = (df['Close'].shift(-days) > df['Close']).astype(int)
    return df


def compute_target_threshold(df: pd.DataFrame, days: int, threshold: float = 0.02) -> pd.DataFrame:
    """
    Compute target with threshold: 1 if price increases > threshold, 0 if decreases > threshold, else 0.5.
    This makes the target less noisy by filtering out small movements.

    Args:
        df: DataFrame with Close column
        days: Number of days ahead
        threshold: Minimum change to count as positive (default 2%)

    Returns:
        DataFrame with 'target' column (0 or 1, filtered by threshold)
    """
    df = df.copy()
    future_return = (df['Close'].shift(-days) - df['Close']) / df['Close']
    df['target'] = (future_return > threshold).astype(int)
    return df


# --- Helper functions ---

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


def _compute_obv(df: pd.DataFrame) -> pd.Series:
    """Compute On Balance Volume."""
    obv = pd.Series(0, index=df.index, dtype=float)
    for i in range(1, len(df)):
        if df['Close'].iloc[i] > df['Close'].iloc[i - 1]:
            obv.iloc[i] = obv.iloc[i - 1] + df['Volume'].iloc[i]
        elif df['Close'].iloc[i] < df['Close'].iloc[i - 1]:
            obv.iloc[i] = obv.iloc[i - 1] - df['Volume'].iloc[i]
        else:
            obv.iloc[i] = obv.iloc[i - 1]
    return obv


def _compute_stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3):
    """Compute Stochastic Oscillator (%K, %D)."""
    low_min = df['Low'].rolling(k_period).min()
    high_max = df['High'].rolling(k_period).max()

    stoch_k = 100 * (df['Close'] - low_min) / (high_max - low_min).replace(0, np.nan)
    stoch_d = stoch_k.rolling(d_period).mean()

    return stoch_k, stoch_d


def _compute_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Compute Average Directional Index (ADX)."""
    high = df['High']
    low = df['Low']
    close = df['Close']

    # +DM and -DM
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

    # True Range
    tr = _compute_atr(df, period=1)  # TR without smoothing

    # Smoothed
    atr = tr.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean() / atr.replace(0, np.nan))
    minus_di = 100 * (minus_dm.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean() / atr.replace(0, np.nan))

    # ADX
    dx = (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan) * 100
    adx = dx.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

    return adx


def _compute_mfi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Compute Money Flow Index (MFI)."""
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    money_flow = typical_price * df['Volume']

    positive_flow = pd.Series(0.0, index=df.index)
    negative_flow = pd.Series(0.0, index=df.index)

    for i in range(1, len(df)):
        if typical_price.iloc[i] > typical_price.iloc[i - 1]:
            positive_flow.iloc[i] = money_flow.iloc[i]
        elif typical_price.iloc[i] < typical_price.iloc[i - 1]:
            negative_flow.iloc[i] = money_flow.iloc[i]

    pos_sum = positive_flow.rolling(period).sum()
    neg_sum = negative_flow.rolling(period).sum()

    mfi = 100 - (100 / (1 + pos_sum / neg_sum.replace(0, np.nan)))
    return mfi


def _compute_williams_r(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Compute Williams %R indicator. Range: -100 to 0."""
    highest_high = df['High'].rolling(period).max()
    lowest_low = df['Low'].rolling(period).min()
    wr = -100 * (highest_high - df['Close']) / (highest_high - lowest_low).replace(0, np.nan)
    return wr


def add_macro_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add macro-economic features (placeholder for future expansion).
    Currently returns the input unchanged.
    """
    # TODO: Add macro-economic indicators (CPI, interest rates, etc.)
    return df
