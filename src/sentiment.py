"""
News Sentiment Analysis / 新聞情緒分析
Fetches news sentiment for HK stocks from AKShare (East Money).
使用 AKShare (東方財富) 獲取港股新聞情緒。

Features / 新增特徵:
- sentiment_5d: 5-day rolling average sentiment / 5天滾動平均情緒
- sentiment_10d: 10-day rolling average sentiment / 10天滾動平均情緒
- sentiment_change: sentiment change (today vs 5d ago) / 情緒變化 (今天 vs 5天前)
"""
import os
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
from src.logger import setup_logger

logger = setup_logger('sentiment')

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(PROJECT_ROOT, 'cache')  # 快取目錄 / Cache directory
os.makedirs(CACHE_DIR, exist_ok=True)

HK_TZ = pytz.timezone('Asia/Hong_Kong')
SENTIMENT_CACHE_TTL = 4 * 3600  # 4 hours / 4小時快取


def _get_cache_path(stock_code: str) -> str:
    """Get cache file path for stock sentiment / 取得股票情緒快取路徑"""
    return os.path.join(CACHE_DIR, f'{stock_code}_sentiment.parquet')


def _is_cache_fresh(cache_path: str) -> bool:
    """Check if cache is still fresh / 檢查快取是否仍然有效"""
    if not os.path.exists(cache_path):
        return False
    age = time.time() - os.path.getmtime(cache_path)
    return age < SENTIMENT_CACHE_TTL


def fetch_sentiment(stock_code: str, days: int = 60) -> pd.DataFrame:
    """
    Fetch news sentiment data for a stock from AKShare (East Money).
    從 AKShare (東方財富) 獲取股票新聞情緒數據。
    
    Returns DataFrame with columns: date, sentiment_score
    sentiment_score: -1 (bearish/看空) to +1 (bullish/看多)
    """
    cache_path = _get_cache_path(stock_code)
    
    # Try loading from cache first / 嘗試從快取載入
    if _is_cache_fresh(cache_path):
        try:
            df = pd.read_parquet(cache_path)
            logger.info(f"[cache] {stock_code}: loaded sentiment from cache / 從快取載入情緒")
            return df
        except Exception as e:
            logger.warning(f"[cache] Failed to load sentiment cache: {e} / 載入快取失敗")
    
    try:
        import akshare as ak
        symbol = str(int(stock_code))
        
        # Fetch news from East Money / 從東方財富獲取新聞
        news_df = ak.stock_news_em(symbol=symbol)
        
        if news_df is None or news_df.empty:
            logger.warning(f"No news data for {stock_code} / 無新聞數據")
            return pd.DataFrame(columns=['date', 'sentiment_score'])
        
        # Parse dates / 解析日期
        date_col = None
        for col in news_df.columns:
            if '日期' in col or 'date' in col.lower():
                date_col = col
                break
        
        if date_col is None:
            date_col = news_df.columns[0]
        
        news_df['date'] = pd.to_datetime(news_df[date_col], errors='coerce')
        news_df = news_df.dropna(subset=['date'])
        
        # Simple keyword-based sentiment scoring / 簡單關鍵字情緒評分
        text_col = None
        for col in news_df.columns:
            if '内容' in col or '标题' in col or 'title' in col.lower() or 'content' in col.lower():
                text_col = col
                break
        
        if text_col is None:
            text_col = news_df.columns[1] if len(news_df.columns) > 1 else news_df.columns[0]
        
        # Bullish keywords / 看多關鍵字
        bullish_words = ['涨', '升', '利好', '看好', '突破', '新高', '增长', '强势', '买入', '推荐']
        # Bearish keywords / 看空關鍵字
        bearish_words = ['跌', '降', '利空', '看空', '破位', '新低', '下滑', '弱势', '卖出', '减持']
        
        sentiments = []
        for _, row in news_df.iterrows():
            text = str(row[text_col])
            bull_count = sum(1 for w in bullish_words if w in text)
            bear_count = sum(1 for w in bearish_words if w in text)
            total = bull_count + bear_count
            if total > 0:
                score = (bull_count - bear_count) / total
            else:
                score = 0.0
            sentiments.append(score)
        
        news_df['sentiment_score'] = sentiments
        news_df = news_df[['date', 'sentiment_score']].copy()
        news_df['date'] = news_df['date'].dt.tz_localize(None).dt.normalize()
        news_df = news_df.sort_values('date').reset_index(drop=True)
        
        # Save cache / 儲存快取
        try:
            news_df.to_parquet(cache_path, index=False)
        except Exception as e:
            logger.warning(f"Failed to save sentiment cache: {e} / 儲存快取失敗")
        
        logger.info(f"[sentiment] {stock_code}: fetched {len(news_df)} news articles / 獲取 {len(news_df)} 篇新聞")
        return news_df
        
    except ImportError:
        logger.warning("akshare not installed, skipping sentiment / akshare 未安裝，跳過情緒分析")
        return pd.DataFrame(columns=['date', 'sentiment_score'])
    except Exception as e:
        logger.warning(f"[sentiment] Failed for {stock_code}: {e} / 情緒分析失敗")
        return pd.DataFrame(columns=['date', 'sentiment_score'])


def compute_sentiment_features(df: pd.DataFrame, stock_code: str) -> pd.DataFrame:
    """
    Add sentiment features to OHLCV DataFrame.
    為 OHLCV DataFrame 新增情緒特徵。
    
    New features / 新增特徵:
    - sentiment_5d: 5-day rolling average sentiment / 5天滾動平均情緒
    - sentiment_10d: 10-day rolling average sentiment / 10天滾動平均情緒
    - sentiment_change: sentiment change (today vs 5d ago) / 情緒變化
    """
    df = df.copy()
    
    sentiment_df = fetch_sentiment(stock_code)
    
    # Default values if no sentiment data / 無情緒數據時使用預設值
    if sentiment_df.empty:
        df['sentiment_5d'] = 0.0
        df['sentiment_10d'] = 0.0
        df['sentiment_change'] = 0.0
        return df
    
    # Merge sentiment by date / 按日期合併情緒數據
    if 'Date' in df.columns:
        df_dates = pd.to_datetime(df['Date']).dt.tz_localize(None).dt.normalize()
    else:
        df_dates = pd.to_datetime(df.index).tz_localize(None).normalize()
    
    sentiment_df = sentiment_df.set_index('date')
    
    # Daily average sentiment / 每日平均情緒
    daily_sentiment = sentiment_df.groupby(sentiment_df.index)['sentiment_score'].mean()
    
    # Map to stock dates / 對應到股票日期
    sentiment_values = []
    for d in df_dates:
        if d in daily_sentiment.index:
            sentiment_values.append(daily_sentiment[d])
        else:
            sentiment_values.append(np.nan)
    
    sentiment_series = pd.Series(sentiment_values, index=df.index)
    
    # Fill NaN with forward fill then 0 / 用前向填充和0填補NaN
    sentiment_series = sentiment_series.ffill().fillna(0.0)
    
    # Compute rolling features / 計算滾動特徵
    df['sentiment_5d'] = sentiment_series.rolling(5, min_periods=1).mean()
    df['sentiment_10d'] = sentiment_series.rolling(10, min_periods=1).mean()
    df['sentiment_change'] = sentiment_series - sentiment_series.shift(5)
    df['sentiment_change'] = df['sentiment_change'].fillna(0.0)
    
    logger.info(f"[sentiment] {stock_code}: added sentiment features / 新增情緒特徵")
    return df


def get_latest_sentiment(stock_code: str) -> dict:
    """Get latest sentiment score for display in dashboard / 取得最新情緒分數供儀表板顯示"""
    sentiment_df = fetch_sentiment(stock_code, days=30)
    if sentiment_df.empty:
        return {'sentiment_score': None, 'sentiment_5d': None}
    
    latest = sentiment_df.iloc[-1]
    avg_5d = sentiment_df.tail(5)['sentiment_score'].mean()
    
    return {
        'sentiment_score': float(latest['sentiment_score']),
        'sentiment_5d': float(avg_5d),
    }
