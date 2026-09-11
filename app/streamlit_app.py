"""
Streamlit dashboard - reads predictions from Supabase and visualizes them.
Supports 3 timeframes: 1d (明日), 5d (下週), 20d (下月)

Dashboard features:
- Latest signal cards with confidence and trend
- Confidence trend chart with interactive threshold line controls
  (select timeframe source, toggle Buy/Sell lines)
- Signal distribution bar chart
- Full prediction table with Buy/Sell thresholds
- CSV/Excel export
- Model metrics, feature importance, monitoring
"""

import os
import sys
import pandas as pd
from datetime import datetime, timedelta
import streamlit as st
from supabase import create_client
from dotenv import load_dotenv
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Load .env from project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

# Page config
st.set_page_config(
    page_title="港股 AI 預測儀表板", layout="wide", initial_sidebar_state="collapsed"
)

# Timeframe labels
TIMEFRAME_LABELS = {
    "1d": "📈 明日 (1天)",
    "5d": "📊 下週 (5天)",
    "20d": "📉 下月 (20天)",
}


# --- Supabase Connection ---
@st.cache_resource
def get_supabase_client():
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    if not url or not key:
        return None
    return create_client(url, key)


@st.cache_data(ttl=300)
def get_predictions(start_date_str, end_date_str):
    client = get_supabase_client()
    if client is None:
        return pd.DataFrame()
    try:
        response = (
            client.table("stock_predictions")
            .select("*")
            .gte("prediction_date", start_date_str)
            .lte("prediction_date", end_date_str)
            .order("prediction_date", desc=True)
            .execute()
        )
        if response.data:
            return pd.DataFrame(response.data)
        return pd.DataFrame()
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=300)
def get_stock_ohlcv(stock_code: str, days: int = 90):
    """Fetch OHLCV data for a stock, cached for 5 minutes."""
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)
    from src.data_fetcher import fetch_stock_data
    try:
        df = fetch_stock_data(stock_code, years=1)
        if df is not None and len(df) > 0:
            # Drop rows with NaN Close (e.g., today's data not yet available)
            df = df.dropna(subset=["Close"])
            return df.tail(days)
    except Exception as e:
        st.warning(f"無法取得 {stock_code} 價格數據: {e}")
    return pd.DataFrame()


@st.cache_data(ttl=300)
def get_latest_indicators(stock_code: str):
    """Compute latest technical indicators for a stock from OHLCV data."""
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)
    from src.data_fetcher import fetch_stock_data
    from src.feature_engineering import compute_features
    try:
        # Need 2 years for reliable indicator calculation (MA50 needs 50+ days)
        df = fetch_stock_data(stock_code, years=2)
        if df is None or len(df) < 60:
            return None
        # Drop rows with NaN Close (e.g., today's data not yet available)
        df = df.dropna(subset=["Close"])
        if len(df) < 60:
            return None
        df = compute_features(df)
        latest = df.iloc[-1]
        return {
            "rsi_14": latest.get("rsi_14"),
            "macd_diff": latest.get("macd_diff"),
            "macd_dea": latest.get("macd_dea"),
            "macd_hist": latest.get("macd_hist"),
            "stoch_k": latest.get("stoch_k"),
            "stoch_d": latest.get("stoch_d"),
            "adx": latest.get("adx"),
            "bb_width": latest.get("bb_width"),
            "atr_14": latest.get("atr_14"),
            "mfi": latest.get("mfi"),
            "vol_ratio_5d": latest.get("vol_ratio_5d"),
            "vol_ratio_10d": latest.get("vol_ratio_10d"),
            "close": latest.get("Close"),
            "ma50_deviation": latest.get("ma50_deviation"),
        }
    except Exception:
        return None


def get_indicator_alignment(indicators: dict, signal: str) -> list:
    """Analyze how current indicators align with a Buy/Sell/Hold signal.
    
    Returns list of (indicator_name, status, description) tuples.
    status: 'support', 'contradict', 'neutral'
    """
    if not indicators:
        return []

    results = []

    # --- RSI ---
    rsi = indicators.get("rsi_14")
    if rsi is not None and pd.notna(rsi):
        if signal == "Buy":
            if rsi < 30:
                results.append(("RSI", "support", f"RSI {rsi:.0f} 超賣，支持反彈上漲"))
            elif rsi < 40:
                results.append(("RSI", "support", f"RSI {rsi:.0f} 偏低，有利買入"))
            elif rsi <= 60:
                results.append(("RSI", "neutral", f"RSI {rsi:.0f} 中性"))
            elif rsi < 70:
                results.append(("RSI", "contradict", f"RSI {rsi:.0f} 偏高，買入需謹慎"))
            else:
                results.append(("RSI", "contradict", f"RSI {rsi:.0f} 超買，不建議追高"))
        elif signal == "Sell":
            if rsi > 70:
                results.append(("RSI", "support", f"RSI {rsi:.0f} 超買，支持回調下跌"))
            elif rsi > 60:
                results.append(("RSI", "support", f"RSI {rsi:.0f} 偏高，有利賣出"))
            elif rsi >= 40:
                results.append(("RSI", "neutral", f"RSI {rsi:.0f} 中性"))
            elif rsi > 30:
                results.append(("RSI", "contradict", f"RSI {rsi:.0f} 偏低，賣出需謹慎"))
            else:
                results.append(("RSI", "contradict", f"RSI {rsi:.0f} 超賣，可能反彈"))
        else:  # Hold
            if rsi < 30:
                results.append(("RSI", "support", f"RSI {rsi:.0f} 超賣，可能反彈"))
            elif rsi > 70:
                results.append(("RSI", "support", f"RSI {rsi:.0f} 超買，可能回調"))
            else:
                results.append(("RSI", "neutral", f"RSI {rsi:.0f} 中性，無明確方向"))

    # --- MACD ---
    macd_hist = indicators.get("macd_hist")
    if macd_hist is not None and pd.notna(macd_hist):
        if signal == "Buy":
            if macd_hist > 0:
                results.append(("MACD", "support", f"MACD 柱狀 {macd_hist:+.4f} 多頭動能"))
            elif macd_hist > -0.01:
                results.append(("MACD", "neutral", f"MACD 柱狀 {macd_hist:+.4f} 接近零軸"))
            else:
                results.append(("MACD", "contradict", f"MACD 柱狀 {macd_hist:+.4f} 空頭動能"))
        elif signal == "Sell":
            if macd_hist < 0:
                results.append(("MACD", "support", f"MACD 柱狀 {macd_hist:+.4f} 空頭動能"))
            elif macd_hist < 0.01:
                results.append(("MACD", "neutral", f"MACD 柱狀 {macd_hist:+.4f} 接近零軸"))
            else:
                results.append(("MACD", "contradict", f"MACD 柱狀 {macd_hist:+.4f} 多頭動能"))
        else:  # Hold
            if abs(macd_hist) < 0.01:
                results.append(("MACD", "neutral", f"MACD 柱狀 {macd_hist:+.4f} 接近零軸，方向不明"))
            elif macd_hist > 0:
                results.append(("MACD", "neutral", f"MACD 柱狀 {macd_hist:+.4f} 輕微多頭"))
            else:
                results.append(("MACD", "neutral", f"MACD 柱狀 {macd_hist:+.4f} 輕微空頭"))

    # --- Stochastic ---
    stoch_k = indicators.get("stoch_k")
    if stoch_k is not None and pd.notna(stoch_k):
        if signal == "Buy":
            if stoch_k < 20:
                results.append(("Stochastic", "support", f"K={stoch_k:.0f} 超賣，支持反彈"))
            elif stoch_k < 40:
                results.append(("Stochastic", "support", f"K={stoch_k:.0f} 偏低"))
            elif stoch_k <= 80:
                results.append(("Stochastic", "neutral", f"K={stoch_k:.0f} 中性"))
            else:
                results.append(("Stochastic", "contradict", f"K={stoch_k:.0f} 超買"))
        elif signal == "Sell":
            if stoch_k > 80:
                results.append(("Stochastic", "support", f"K={stoch_k:.0f} 超買，支持回調"))
            elif stoch_k > 60:
                results.append(("Stochastic", "support", f"K={stoch_k:.0f} 偏高"))
            elif stoch_k >= 20:
                results.append(("Stochastic", "neutral", f"K={stoch_k:.0f} 中性"))
            else:
                results.append(("Stochastic", "contradict", f"K={stoch_k:.0f} 超賣"))
        else:  # Hold
            if stoch_k < 20:
                results.append(("Stochastic", "support", f"K={stoch_k:.0f} 超賣，可能反彈"))
            elif stoch_k > 80:
                results.append(("Stochastic", "support", f"K={stoch_k:.0f} 超買，可能回調"))
            else:
                results.append(("Stochastic", "neutral", f"K={stoch_k:.0f} 中性"))

    # --- MFI ---
    mfi = indicators.get("mfi")
    if mfi is not None and pd.notna(mfi):
        if signal == "Buy":
            if mfi < 20:
                results.append(("MFI", "support", f"MFI {mfi:.0f} 資金枯竭，可能反彈"))
            elif mfi < 40:
                results.append(("MFI", "support", f"MFI {mfi:.0f} 資金偏弱"))
            elif mfi <= 60:
                results.append(("MFI", "neutral", f"MFI {mfi:.0f} 中性"))
            else:
                results.append(("MFI", "contradict", f"MFI {mfi:.0f} 資金偏強"))
        elif signal == "Sell":
            if mfi > 80:
                results.append(("MFI", "support", f"MFI {mfi:.0f} 資金過熱，可能回調"))
            elif mfi > 60:
                results.append(("MFI", "support", f"MFI {mfi:.0f} 資金偏強"))
            elif mfi >= 40:
                results.append(("MFI", "neutral", f"MFI {mfi:.0f} 中性"))
            else:
                results.append(("MFI", "contradict", f"MFI {mfi:.0f} 資金偏弱"))
        else:  # Hold
            if mfi < 20:
                results.append(("MFI", "support", f"MFI {mfi:.0f} 資金枯竭，可能反彈"))
            elif mfi > 80:
                results.append(("MFI", "support", f"MFI {mfi:.0f} 資金過熱，可能回調"))
            else:
                results.append(("MFI", "neutral", f"MFI {mfi:.0f} 中性"))

    # --- ADX ---
    adx = indicators.get("adx")
    if adx is not None and pd.notna(adx):
        if adx > 25:
            # Strong trend - supports following the signal direction
            if signal == "Hold":
                results.append(("ADX", "contradict", f"ADX {adx:.0f} 趨勢明確，但信號為 Hold"))
            else:
                results.append(("ADX", "support", f"ADX {adx:.0f} 趨勢明確，支持跟隨方向"))
        elif adx > 20:
            results.append(("ADX", "neutral", f"ADX {adx:.0f} 趨勢不明確"))
        else:
            results.append(("ADX", "contradict", f"ADX {adx:.0f} 盤整，信號可靠性降低"))

    # --- BB Width ---
    bb_width = indicators.get("bb_width")
    if bb_width is not None and pd.notna(bb_width):
        if bb_width < 0.03:
            results.append(("布林帶", "neutral", f"帶寬 {bb_width:.4f} 極窄，可能即將變盤"))
        elif bb_width > 0.15:
            results.append(("布林帶", "contradict", f"帶寬 {bb_width:.4f} 較大，波動性高"))
        else:
            results.append(("布林帶", "neutral", f"帶寬 {bb_width:.4f} 正常"))

    # --- MA50 Deviation ---
    ma50_dev = indicators.get("ma50_deviation")
    if ma50_dev is not None and pd.notna(ma50_dev):
        ma50_pct = ma50_dev * 100  # Convert decimal to percentage
        if signal == "Buy":
            if ma50_pct < -5:
                results.append(("MA50", "support", f"偏離 {ma50_pct:+.1f}% 低於均線，可能回歸"))
            elif ma50_pct < 0:
                results.append(("MA50", "support", f"偏離 {ma50_pct:+.1f}% 在均線下方"))
            elif ma50_pct < 5:
                results.append(("MA50", "neutral", f"偏離 {ma50_pct:+.1f}% 接近均線"))
            else:
                results.append(("MA50", "contradict", f"偏離 {ma50_pct:+.1f}% 已高於均線"))
        elif signal == "Sell":
            if ma50_pct > 5:
                results.append(("MA50", "support", f"偏離 {ma50_pct:+.1f}% 高於均線，可能回落"))
            elif ma50_pct > 0:
                results.append(("MA50", "support", f"偏離 {ma50_pct:+.1f}% 在均線上方"))
            elif ma50_pct > -5:
                results.append(("MA50", "neutral", f"偏離 {ma50_pct:+.1f}% 接近均線"))
            else:
                results.append(("MA50", "contradict", f"偏離 {ma50_pct:+.1f}% 已低於均線"))
        else:  # Hold
            if abs(ma50_pct) < 2:
                results.append(("MA50", "neutral", f"偏離 {ma50_pct:+.1f}% 接近均線"))
            elif ma50_pct < 0:
                results.append(("MA50", "support", f"偏離 {ma50_pct:+.1f}% 在均線下方，可能回歸"))
            else:
                results.append(("MA50", "support", f"偏離 {ma50_pct:+.1f}% 在均線上方，可能回落"))

    return results


# --- Main Page ---
st.title("📈 港股 AI 預測儀表板")

# Check connection
client = get_supabase_client()
if client is None:
    st.error("無法連線至資料庫。請檢查 .env 中的 SUPABASE_URL 和 SUPABASE_KEY。")
    st.stop()

# --- Sidebar Controls ---
st.sidebar.subheader("📅 日期範圍")
col_start, col_end = st.sidebar.columns(2)
with col_start:
    start_date = st.date_input(
        "開始日期", value=(datetime.now() - timedelta(days=31)).date()
    )
with col_end:
    end_date = st.date_input(
        "結束日期", value=(datetime.now() + timedelta(days=31)).date()
    )

# Clear cache to force fresh data on date change
df = get_predictions(start_date.isoformat(), end_date.isoformat())

if df.empty:
    st.info(
        "暫無預測數據。請先執行:\n1. `python src/train_model.py` 訓練模型\n2. `python src/predict_upload.py` 生成預測"
    )
    st.stop()

# Ensure proper types
df["prediction_date"] = pd.to_datetime(df["prediction_date"]).dt.date
df["confidence"] = df["confidence"].astype(float)
df["stock_code"] = df["stock_code"].astype(str)
if "timeframe" not in df.columns:
    df["timeframe"] = "1d"  # Legacy data


# --- Tab Layout ---
tab_signals, tab_kline, tab_confidence, tab_history, tab_performance = st.tabs([
    "📊 信號總覽", "📈 K線與指標", "🎯 信心度趨勢", "📋 預測記錄", "🔍 模型表現"
])

with tab_signals:
    # --- Latest Signals for Each Timeframe ---
    st.markdown("---")

    for tf_label, tf_title in TIMEFRAME_LABELS.items():
        st.subheader(tf_title)

        tf_df = df[df["timeframe"] == tf_label]
        if tf_df.empty:
            st.info(f"暫無 {tf_label} 預測數據")
            continue

        # Get latest per stock
        latest = (
            tf_df.sort_values("prediction_date").groupby("stock_code").last().reset_index()
        )

        # Show threshold info for this timeframe (use first stock with valid thresholds as reference)
        if "threshold_buy" in latest.columns and latest["threshold_buy"].notna().any():
            ref = latest[latest["threshold_buy"].notna()].iloc[0] if not latest[latest["threshold_buy"].notna()].empty else None
            if ref is not None:
                st.caption(
                    f"信心 = 模型預測上漲的機率 | Buy 閾值: {ref['threshold_buy']:.0%} | Sell 閾值: {ref['threshold_sell']:.0%}"
                )
            else:
                st.caption("信心 = 模型預測上漲的機率 | Buy 閾值: 55% | Sell 閾值: 45% (預設)")
        else:
            st.caption("信心 = 模型預測上漲的機率 | Buy 閾值: 55% | Sell 閾值: 45% (預設)")

        cols = st.columns(len(latest))
        for i, (_, row) in enumerate(latest.iterrows()):
            with cols[i]:
                signal = row["signal"]
                if signal == "Buy":
                    emoji = "📈"
                    delta_color = "normal"
                elif signal == "Sell":
                    emoji = "📉"
                    delta_color = "inverse"
                else:
                    emoji = "➡️"
                    delta_color = "off"

                # Build delta with threshold context
                conf = row["confidence"]
                if "threshold_buy" in row and pd.notna(row["threshold_buy"]):
                    buy_th = row["threshold_buy"]
                    sell_th = row["threshold_sell"]
                    delta_text = f"信心: {conf:.1%} | Buy>{buy_th:.0%} Sell<{sell_th:.0%}"
                else:
                    delta_text = f"信心: {conf:.1%}"

                # Add model disagreement indicator
                if "model_disagreement" in row and pd.notna(row["model_disagreement"]):
                    disagreement = row["model_disagreement"]
                    model_split = row.get("model_split", "?/?")
                    if disagreement >= 0.5:
                        delta_text += f" ⚠️ 分歧: {model_split}"
                    elif disagreement > 0:
                        delta_text += f" | 分歧: {model_split}"

                st.metric(
                    label=f"{emoji} {row['stock_code']}",
                    value=signal,
                    delta=delta_text,
                    delta_color=delta_color,
                )

                # --- Indicator Alignment Analysis ---
                stock_indicators = get_latest_indicators(row["stock_code"])
                if stock_indicators:
                    alignment = get_indicator_alignment(stock_indicators, signal)
                    if alignment:
                        supports = sum(1 for _, s, _ in alignment if s == "support")
                        contradicts = sum(1 for _, s, _ in alignment if s == "contradict")
                        neutrals = sum(1 for _, s, _ in alignment if s == "neutral")

                        # Summary line
                        parts = []
                        if supports:
                            parts.append(f"{supports} 項支持")
                        if neutrals:
                            parts.append(f"{neutrals} 項中性")
                        if contradicts:
                            parts.append(f"{contradicts} 項矛盾")
                        summary = ", ".join(parts)

                        if signal == "Hold":
                            # For Hold: show why model chose Hold instead of Buy/Sell
                            if contradicts > 0 and supports > 0:
                                st.caption(f"⚠️ 指標分析: {summary} (多空分歧，模型選擇觀望)")
                            elif contradicts > supports:
                                st.caption(f"⚠️ 指標分析: {summary} (指標矛盾，建議觀望)")
                            else:
                                st.caption(f"➖ 指標分析: {summary} (無明確方向)")
                        elif contradicts > supports:
                            st.caption(f"⚠️ 指標分析: {summary}")
                        elif supports > 0:
                            st.caption(f"✅ 指標分析: {summary}")
                        else:
                            st.caption(f"➖ 指標分析: {summary}")

                        # Per-indicator details (always show when alignment exists)
                        for name, status, desc in alignment:
                            if status == "support":
                                icon = "✅"
                            elif status == "contradict":
                                icon = "❌"
                            else:
                                icon = "➖"
                            st.caption(f"  {icon} {name} → {desc}")

        st.markdown("---")

with tab_kline:
    # --- Technical Indicators (Supporting Evidence) ---
    st.subheader("🔬 技術指標 (支持信號依據)")
    st.caption(
        "以下指標為模型訓練時輸入的 33 項特徵中的關鍵項目。"
        "同一組指標同時用於預測 **1天 (1d)、5天 (5d)、20天 (20d)** 三個時間範圍的 Buy/Sell 信號。"
        "信心度 = 模型根據這些指標學習到的模式所給出的機率。"
    )

    # Get all unique stock codes from current predictions
    indicator_stocks = sorted(df["stock_code"].unique())
    selected_indicator_stock = st.selectbox(
        "選擇股票查看指標",
        options=indicator_stocks,
        key="indicator_stock",
    )

    if selected_indicator_stock:
        indicators = get_latest_indicators(selected_indicator_stock)

        if indicators:
            # Row 1: Momentum indicators
            st.markdown("**📈 動量指標**")
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                rsi = indicators.get("rsi_14")
                if rsi is not None and pd.notna(rsi):
                    rsi_color = "🟢" if 40 <= rsi <= 60 else ("🔴" if rsi > 70 or rsi < 30 else "🟡")
                    st.metric(
                        f"{rsi_color} RSI (14)",
                        f"{rsi:.1f}",
                        help=">70 超買 | <30 超賣 | 40-60 中性",
                    )
                else:
                    st.metric("RSI (14)", "—", help="需至少 14 天收盤價數據。RSI = 100 - 100/(1+RS)，衡量價格動量。")

            with col2:
                stoch_k = indicators.get("stoch_k")
                stoch_d = indicators.get("stoch_d")
                if stoch_k is not None and stoch_d is not None and pd.notna(stoch_k) and pd.notna(stoch_d):
                    stoch_color = "🟢" if 20 <= stoch_k <= 80 else ("🔴" if stoch_k > 80 or stoch_k < 20 else "🟡")
                    st.metric(
                        f"{stoch_color} Stochastic K/D",
                        f"{stoch_k:.1f} / {stoch_d:.1f}",
                        help=">80 超買 | <20 超賣",
                    )
                else:
                    st.metric("Stochastic K/D", "—", help="需至少 17 天數據 (14天窗口+3天D線平滑)。計算最近14天內收盤價相對高低點的位置。")

            with col3:
                adx = indicators.get("adx")
                if adx is not None and pd.notna(adx):
                    adx_label = "強趨勢" if adx > 25 else "弱趨勢/盤整"
                    adx_color = "🟢" if adx > 25 else "🟡"
                    st.metric(
                        f"{adx_color} ADX",
                        f"{adx:.1f}",
                        delta=adx_label,
                        help=">25 趨勢明確 | <20 盤整",
                    )
                else:
                    st.metric("ADX", "—", help="需至少 28 天數據 (14天×2 次平滑)。衡量趨勢強度，不分方向。")

            with col4:
                mfi = indicators.get("mfi")
                if mfi is not None and pd.notna(mfi):
                    mfi_color = "🟢" if 40 <= mfi <= 60 else ("🔴" if mfi > 80 or mfi < 20 else "🟡")
                    st.metric(
                        f"{mfi_color} MFI",
                        f"{mfi:.1f}",
                        help=">80 資金過熱 | <20 資金枯竭",
                    )
                else:
                    st.metric("MFI", "—", help="需至少 14 天 OHLCV 數據。結合價格和成交量計算資金流入/流出。")

            # Row 2: Trend indicators
            st.markdown("**📊 趨勢指標**")
            col5, col6, col7, col8 = st.columns(4)

            with col5:
                macd_hist = indicators.get("macd_hist")
                if macd_hist is not None and pd.notna(macd_hist):
                    macd_signal = "📈 多頭" if macd_hist > 0 else "📉 空頭"
                    macd_color = "normal" if macd_hist > 0 else "inverse"
                    st.metric(
                        "MACD 柱狀",
                        f"{macd_hist:+.4f}",
                        delta=macd_signal,
                        delta_color=macd_color,
                        help="正=多頭動能 | 負=空頭動能",
                    )
                else:
                    st.metric("MACD 柱狀", "—", help="需至少 34 天數據 (EMA26+信號線9)。MACD = 12日EMA - 26日EMA。")

            with col6:
                bb_width = indicators.get("bb_width")
                if bb_width is not None and pd.notna(bb_width):
                    st.metric(
                        "布林帶寬",
                        f"{bb_width:.4f}",
                        help="寬=波動大 | 窄=波動小（可能變盤）",
                    )
                else:
                    st.metric("布林帶寬", "—", help="需至少 20 天收盤價。布林帶寬 = (上軌-下軌)/中軌，衡量波動率。")

            with col7:
                atr = indicators.get("atr_14")
                close = indicators.get("close")
                if atr is not None and close is not None and pd.notna(atr) and pd.notna(close) and close > 0:
                    atr_pct = (atr / close) * 100
                    st.metric(
                        "ATR (14)",
                        f"{atr:.2f}",
                        delta=f"{atr_pct:.2f}% of price",
                        help="平均真實波幅，衡量日內波動",
                    )
                else:
                    st.metric("ATR (14)", "—", help="需至少 14 天 OHLC 數據。ATR = 最近14天真實波幅的指數移動平均。")

            with col8:
                ma50_dev = indicators.get("ma50_deviation")
                if ma50_dev is not None and pd.notna(ma50_dev):
                    ma50_pct = ma50_dev * 100  # Convert decimal to percentage
                    ma50_label = "偏離均線" if abs(ma50_pct) > 5 else "接近均線"
                    ma50_color = "normal" if ma50_dev > 0 else "inverse"
                    st.metric(
                        "MA50 偏離",
                        f"{ma50_pct:+.2f}%",
                        delta=ma50_label,
                        delta_color=ma50_color,
                        help="正=價格在均線上方 | 負=價格在均線下方",
                    )
                else:
                    st.metric("MA50 偏離", "—", help="需至少 50 天收盤價。偏離率 = (現價-50日均線)/50日均線。")

            # Row 3: Volume indicators
            st.markdown("**📦 成交量指標**")
            col9, col10 = st.columns(2)

            with col9:
                vol_5d = indicators.get("vol_ratio_5d")
                if vol_5d is not None and pd.notna(vol_5d):
                    vol_label = "量能放大" if vol_5d > 1.2 else ("量能萎縮" if vol_5d < 0.8 else "量能正常")
                    vol_color = "normal" if vol_5d > 1.2 else ("inverse" if vol_5d < 0.8 else "off")
                    st.metric(
                        "5日量比",
                        f"{vol_5d:.2f}x",
                        delta=vol_label,
                        delta_color=vol_color,
                        help=">1.2 放量 | <0.8 縮量",
                    )
                else:
                    st.metric("5日量比", "—", help="需至少 5 天成交量數據。量比 = 今日成交量 / 5日平均成交量。")

            with col10:
                vol_10d = indicators.get("vol_ratio_10d")
                if vol_10d is not None and pd.notna(vol_10d):
                    st.metric(
                        "10日量比",
                        f"{vol_10d:.2f}x",
                        help="10日平均成交量相對比率",
                    )
                else:
                    st.metric("10日量比", "—", help="需至少 10 天成交量數據。量比 = 今日成交量 / 10日平均成交量。")

            # Indicator interpretation
            with st.expander("📖 指標解讀說明"):
                st.markdown("""
    | 指標 | 正常範圍 | 超買/超賣 | 說明 |
    |---|---|---|---|
    | **RSI (14)** | 40-60 | >70 超買 / <30 超賣 | 相對強弱指標，衡量價格動量 |
    | **Stochastic K/D** | 20-80 | >80 超買 / <20 超賣 | 隨機震盪指標，判斷超買超賣 |
    | **ADX** | >25 為趨勢 | <20 盤整 | 趨勢強度（不分方向） |
    | **MFI** | 40-60 | >80 過熱 / <20 枯竭 | 資金流量指標，結合價格和成交量 |
    | **MACD 柱狀** | 正=多頭 | 負=空頭 | 多空動能指標 |
    | **布林帶寬** | 視情況 | 極窄=變盤 | 價格波動範圍 |
    | **ATR** | 視股價 | 高=波動大 | 平均真實波幅 |
    | **MA50 偏離** | ±5% 內 | >10% 偏離 | 與50日均線乖離率 |
    | **量比** | 0.8-1.2 | >1.5 放量 | 成交量相對強弱 |
                """)
        else:
            st.info(f"無法載入 {selected_indicator_stock} 的指標數據。")

with tab_confidence:
    # --- Confidence Trend (all timeframes) ---
    st.subheader("📊 預測信心度趨勢")

    # Threshold toggle controls
    col_tf, col_show_buy, col_show_sell = st.columns(3)
    with col_tf:
        th_options = (
            ["不顯示"] + list(df["timeframe"].unique()) if not df.empty else ["不顯示"]
        )
        selected_th_tf = st.selectbox("閾值來源時間範圍", th_options, key="th_timeframe")
    with col_show_buy:
        show_buy = st.checkbox("顯示 Buy 閾值線", value=True, key="show_buy_th")
    with col_show_sell:
        show_sell = st.checkbox("顯示 Sell 閾值線", value=True, key="show_sell_th")

    fig_trend = px.line(
        df,
        x="prediction_date",
        y="confidence",
        color="stock_code",
        symbol="timeframe",
        markers=True,
        labels={
            "prediction_date": "日期",
            "confidence": "信心度",
            "stock_code": "股票代碼",
            "timeframe": "時間範圍",
        },
        title="各股票預測信心度變化",
    )

    # Plot threshold lines from database (per day, per timeframe)
    if (
        selected_th_tf != "不顯示"
        and "threshold_buy" in df.columns
        and df["threshold_buy"].notna().any()
    ):
        th_df = df[df["timeframe"] == selected_th_tf].sort_values("prediction_date")
        if show_buy and th_df["threshold_buy"].notna().any():
            fig_trend.add_trace(
                go.Scatter(
                    x=th_df["prediction_date"],
                    y=th_df["threshold_buy"],
                    mode="lines",
                    name=f"Buy 閾值 ({selected_th_tf})",
                    line=dict(color="green", dash="dash", width=2),
                    hovertemplate="Buy 閾值: %{y:.0%}<extra></extra>",
                )
            )
        if show_sell and th_df["threshold_sell"].notna().any():
            fig_trend.add_trace(
                go.Scatter(
                    x=th_df["prediction_date"],
                    y=th_df["threshold_sell"],
                    mode="lines",
                    name=f"Sell 閾值 ({selected_th_tf})",
                    line=dict(color="red", dash="dash", width=2),
                    hovertemplate="Sell 閾值: %{y:.0%}<extra></extra>",
                )
            )
    elif selected_th_tf != "不顯示":
        # Fallback: single horizontal lines
        if show_buy:
            fig_trend.add_hline(
                y=0.55, line_dash="dash", line_color="green", annotation_text="Buy (55%)"
            )
        if show_sell:
            fig_trend.add_hline(
                y=0.45, line_dash="dash", line_color="red", annotation_text="Sell (45%)"
            )

    fig_trend.update_layout(yaxis_tickformat=".0%")
    st.plotly_chart(fig_trend, use_container_width=True)

with tab_kline:
    # --- K-Line Chart with Buy/Sell Signals ---
    st.markdown("---")
    st.subheader("🕯️ K線圖 (含買賣信號)")
    st.caption("互動式K線圖，標示模型預測的買入/賣出信號位置，支持MA均線 (MA5/10/20)")

    # Stock selector for K-line chart
    stock_codes_in_df = sorted(df["stock_code"].unique())
    col_stock, col_period, col_tf = st.columns(3)
    with col_stock:
        selected_kline_stock = st.selectbox(
            "選擇股票",
            options=stock_codes_in_df,
            key="kline_stock",
        )
    with col_period:
        kline_period = st.selectbox(
            "顯示天數",
            options=[30, 60, 90, 180],
            index=2,  # Default 90 days
            format_func=lambda x: f"{x} 天",
            key="kline_period",
        )
    with col_tf:
        kline_tf = st.selectbox(
            "預測時間範圍",
            options=["all", "1d", "5d", "20d"],
            format_func=lambda x: "全部" if x == "all" else TIMEFRAME_LABELS.get(x, x),
            key="kline_tf",
        )

    # MA overlay checkboxes
    col_ma5, col_ma10, col_ma20 = st.columns(3)
    with col_ma5:
        show_ma5 = st.checkbox("MA5", value=True, key="chk_ma5")
    with col_ma10:
        show_ma10 = st.checkbox("MA10", value=True, key="chk_ma10")
    with col_ma20:
        show_ma20 = st.checkbox("MA20", value=True, key="chk_ma20")

    if selected_kline_stock:
        ohlcv_df = get_stock_ohlcv(selected_kline_stock, days=kline_period)

        if not ohlcv_df.empty:
            # Drop rows with NaN Close prices
            ohlcv_df = ohlcv_df.dropna(subset=["Close"])

            # Calculate moving averages
            if show_ma5:
                ohlcv_df["MA5"] = ohlcv_df["Close"].rolling(window=5).mean()
            if show_ma10:
                ohlcv_df["MA10"] = ohlcv_df["Close"].rolling(window=10).mean()
            if show_ma20:
                ohlcv_df["MA20"] = ohlcv_df["Close"].rolling(window=20).mean()

            # Get predictions for this stock within the OHLCV date range
            min_date = ohlcv_df["Date"].min().date()
            max_date = ohlcv_df["Date"].max().date()
            stock_preds = df[
                (df["stock_code"] == selected_kline_stock)
                & (df["prediction_date"] >= min_date)
                & (df["prediction_date"] <= max_date)
            ].copy()

            # Filter by timeframe if selected
            if kline_tf != "all":
                stock_preds = stock_preds[stock_preds["timeframe"] == kline_tf]

            # Create candlestick chart
            tf_label = "全部" if kline_tf == "all" else TIMEFRAME_LABELS.get(kline_tf, kline_tf)
            fig_kline = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.03,
                row_heights=[0.7, 0.3],
                subplot_titles=(f"{selected_kline_stock} K線圖 — {tf_label} 預測", "成交量"),
            )

            # Candlestick
            fig_kline.add_trace(
                go.Candlestick(
                    x=ohlcv_df["Date"],
                    open=ohlcv_df["Open"],
                    high=ohlcv_df["High"],
                    low=ohlcv_df["Low"],
                    close=ohlcv_df["Close"],
                    name="K線",
                    increasing_line_color="#26a69a",
                    decreasing_line_color="#ef5350",
                ),
                row=1, col=1,
            )

            # Add MA lines
            ma_colors = {"MA5": "#FF6B6B", "MA10": "#4ECDC4", "MA20": "#45B7D1"}
            for ma_col, ma_color in ma_colors.items():
                if ma_col in ohlcv_df.columns:
                    fig_kline.add_trace(
                        go.Scatter(
                            x=ohlcv_df["Date"],
                            y=ohlcv_df[ma_col],
                            name=ma_col,
                            line=dict(color=ma_color, width=1.5),
                            opacity=0.8,
                        ),
                        row=1, col=1,
                    )

            # Volume bars
            colors = [
                "#26a69a" if c >= o else "#ef5350"
                for c, o in zip(ohlcv_df["Close"], ohlcv_df["Open"])
            ]
            fig_kline.add_trace(
                go.Bar(
                    x=ohlcv_df["Date"],
                    y=ohlcv_df["Volume"],
                    name="成交量",
                    marker_color=colors,
                    opacity=0.6,
                ),
                row=2, col=1,
            )

            # Add Buy/Sell signal markers (grouped by timeframe)
            if not stock_preds.empty and "signal" in stock_preds.columns:
                # Marker config per timeframe: symbol, size
                tf_markers = {
                    "1d": ("circle", 10),
                    "5d": ("square", 12),
                    "20d": ("diamond", 14),
                }
                tf_labels = {"1d": "明日", "5d": "下週", "20d": "下月"}

                for tf_key, (symbol, size) in tf_markers.items():
                    tf_label = tf_labels.get(tf_key, tf_key)
                    tf_preds = stock_preds[stock_preds["timeframe"] == tf_key]
                    if tf_preds.empty:
                        continue

                    # Buy signals for this timeframe
                    buy_tf = tf_preds[tf_preds["signal"] == "Buy"]
                    if not buy_tf.empty:
                        buy_dates = pd.to_datetime(buy_tf["prediction_date"])
                        buy_y = []
                        for d in buy_dates:
                            match = ohlcv_df[ohlcv_df["Date"] == d]
                            if not match.empty:
                                buy_y.append(match.iloc[0]["Low"] * 0.98)
                            else:
                                buy_y.append(None)
                        buy_tf = buy_tf.copy()
                        buy_tf["_plot_y"] = buy_y
                        buy_tf = buy_tf.dropna(subset=["_plot_y"])

                        if not buy_tf.empty:
                            fig_kline.add_trace(
                                go.Scatter(
                                    x=buy_tf["prediction_date"],
                                    y=buy_tf["_plot_y"],
                                    mode="markers",
                                    name=f"📈 Buy ({tf_label})",
                                    marker=dict(
                                        symbol=symbol,
                                        size=size,
                                        color="#2ecc71",
                                        line=dict(width=2, color="darkgreen"),
                                    ),
                                    text=buy_tf.apply(
                                        lambda r: f"Buy {tf_label}<br>信心: {r['confidence']:.1%}", axis=1
                                    ),
                                    hovertemplate="%{text}<br>%{x}<extra></extra>",
                                ),
                                row=1, col=1,
                            )

                    # Sell signals for this timeframe
                    sell_tf = tf_preds[tf_preds["signal"] == "Sell"]
                    if not sell_tf.empty:
                        sell_dates = pd.to_datetime(sell_tf["prediction_date"])
                        sell_y = []
                        for d in sell_dates:
                            match = ohlcv_df[ohlcv_df["Date"] == d]
                            if not match.empty:
                                sell_y.append(match.iloc[0]["High"] * 1.02)
                            else:
                                sell_y.append(None)
                        sell_tf = sell_tf.copy()
                        sell_tf["_plot_y"] = sell_y
                        sell_tf = sell_tf.dropna(subset=["_plot_y"])

                        if not sell_tf.empty:
                            fig_kline.add_trace(
                                go.Scatter(
                                    x=sell_tf["prediction_date"],
                                    y=sell_tf["_plot_y"],
                                    mode="markers",
                                    name=f"📉 Sell ({tf_label})",
                                    marker=dict(
                                        symbol=symbol,
                                        size=size,
                                        color="#e74c3c",
                                        line=dict(width=2, color="darkred"),
                                    ),
                                    text=sell_tf.apply(
                                        lambda r: f"Sell {tf_label}<br>信心: {r['confidence']:.1%}", axis=1
                                    ),
                                    hovertemplate="%{text}<br>%{x}<extra></extra>",
                                ),
                                row=1, col=1,
                            )

            fig_kline.update_layout(
                height=600,
                xaxis_rangeslider_visible=False,
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            fig_kline.update_xaxes(title_text="", row=2, col=1)
            fig_kline.update_yaxes(title_text="價格 (HKD)", row=1, col=1)
            fig_kline.update_yaxes(title_text="成交量", row=2, col=1)

            st.plotly_chart(fig_kline, use_container_width=True)

            # Summary stats
            latest_close = ohlcv_df.iloc[-1]["Close"]
            prev_close = ohlcv_df.iloc[-2]["Close"] if len(ohlcv_df) > 1 else latest_close

            # Guard against NaN
            if pd.isna(latest_close) or pd.isna(prev_close) or prev_close == 0:
                daily_change = None
            else:
                daily_change = (latest_close - prev_close) / prev_close

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                if pd.notna(latest_close):
                    delta_str = f"{daily_change:+.2%}" if daily_change is not None else ""
                    st.metric("最新收盤", f"HKD {latest_close:.2f}", delta_str)
                else:
                    st.metric("最新收盤", "—", help="無法取得最新收盤價，可能是非交易日或數據缺失")
            with col2:
                high_max = ohlcv_df["High"].max()
                st.metric(
                    "最高價",
                    f"HKD {high_max:.2f}" if pd.notna(high_max) else "—",
                    help="所選時間範圍內的最高價" if pd.notna(high_max) else "數據不足，無法計算最高價",
                )
            with col3:
                low_min = ohlcv_df["Low"].min()
                st.metric(
                    "最低價",
                    f"HKD {low_min:.2f}" if pd.notna(low_min) else "—",
                    help="所選時間範圍內的最低價" if pd.notna(low_min) else "數據不足，無法計算最低價",
                )
            with col4:
                buy_count = len(stock_preds[stock_preds["signal"] == "Buy"]) if not stock_preds.empty else 0
                sell_count = len(stock_preds[stock_preds["signal"] == "Sell"]) if not stock_preds.empty else 0
                st.metric("信號統計", f"Buy: {buy_count} / Sell: {sell_count}")
        else:
            st.info(f"無法載入 {selected_kline_stock} 的價格數據，請稍後再試。")

with tab_kline:
    # --- Stock Comparison View ---
    st.subheader("📊 股票對比")
    st.caption("選擇兩支股票並排比較價格走勢和表現")

    col_cmp1, col_cmp2, col_cmp_period = st.columns(3)
    with col_cmp1:
        cmp_stock1 = st.selectbox(
            "股票 A",
            options=stock_codes_in_df,
            index=0,
            key="cmp_stock1",
        )
    with col_cmp2:
        cmp_stock2 = st.selectbox(
            "股票 B",
            options=stock_codes_in_df,
            index=min(1, len(stock_codes_in_df) - 1),
            key="cmp_stock2",
        )
    with col_cmp_period:
        cmp_period = st.selectbox(
            "顯示天數",
            options=[30, 60, 90, 180],
            index=2,
            format_func=lambda x: f"{x} 天",
            key="cmp_period",
        )

    if cmp_stock1 and cmp_stock2:
        df_cmp1 = get_stock_ohlcv(cmp_stock1, days=cmp_period)
        df_cmp2 = get_stock_ohlcv(cmp_stock2, days=cmp_period)

        if not df_cmp1.empty and not df_cmp2.empty:
            # Normalize to percentage change for comparison
            df_cmp1 = df_cmp1.dropna(subset=["Close"]).copy()
            df_cmp2 = df_cmp2.dropna(subset=["Close"]).copy()

            df_cmp1["Return_%"] = (df_cmp1["Close"] / df_cmp1["Close"].iloc[0] - 1) * 100
            df_cmp2["Return_%"] = (df_cmp2["Close"] / df_cmp2["Close"].iloc[0] - 1) * 100

            fig_cmp = go.Figure()
            fig_cmp.add_trace(go.Scatter(
                x=df_cmp1["Date"], y=df_cmp1["Return_%"],
                name=cmp_stock1, line=dict(width=2),
            ))
            fig_cmp.add_trace(go.Scatter(
                x=df_cmp2["Date"], y=df_cmp2["Return_%"],
                name=cmp_stock2, line=dict(width=2),
            ))
            fig_cmp.update_layout(
                title=f"{cmp_stock1} vs {cmp_stock2} — 累計報酬率 (%)",
                yaxis_title="累計報酬 (%)",
                height=400,
                showlegend=True,
            )
            st.plotly_chart(fig_cmp, use_container_width=True)

            # Side-by-side metrics
            m1_col, m2_col = st.columns(2)
            for col, sym, d in [(m1_col, cmp_stock1, df_cmp1), (m2_col, cmp_stock2, df_cmp2)]:
                with col:
                    latest = d.iloc[-1]["Close"]
                    first = d.iloc[0]["Close"]
                    ret = (latest / first - 1) * 100
                    vol = d["Volume"].mean()
                    high = d["High"].max()
                    low = d["Low"].min()
                    st.markdown(f"**{sym}**")
                    st.write(f"最新: HKD {latest:.2f} | 報酬: {ret:+.2f}%")
                    st.write(f"最高: HKD {high:.2f} | 最低: HKD {low:.2f}")
                    st.write(f"平均成交量: {vol:,.0f}")
        else:
            st.info("無法載入股票數據進行對比。")

with tab_signals:
    # --- Signal Distribution per Timeframe ---
    st.subheader("📋 信號分佈")
    selected_tf = st.selectbox(
        "選擇時間範圍",
        list(TIMEFRAME_LABELS.keys()),
        format_func=lambda x: TIMEFRAME_LABELS[x],
    )

    tf_df = df[df["timeframe"] == selected_tf]
    if not tf_df.empty:
        signal_counts = (
            tf_df.groupby(["stock_code", "signal"]).size().reset_index(name="count")
        )
        signal_counts["stock_code"] = signal_counts["stock_code"].astype(str)
        fig_pie = px.bar(
            signal_counts,
            x="stock_code",
            y="count",
            color="signal",
            color_discrete_map={"Buy": "#2ecc71", "Sell": "#e74c3c", "Hold": "#95a5a6"},
            labels={"stock_code": "股票代碼", "count": "次數", "signal": "信號"},
            title=f"{TIMEFRAME_LABELS[selected_tf]} 信號分佈",
        )
        fig_pie.update_xaxes(type="category")
        st.plotly_chart(fig_pie, use_container_width=True)

with tab_history:
    # --- Recent Predictions Table ---
    st.markdown("---")
    st.subheader("📝 近期預測記錄")

    # Only select known columns, ignore extras (id, created_at, etc.)
    base_cols = [
        "stock_code",
        "prediction_date",
        "timeframe",
        "signal",
        "confidence",
        "model_version",
        "created_at",
    ]
    extra_cols = [
        "model_type",
        "f1_score",
        "auc_score",
        "expected_return",
        "risk_reward",
        "stop_loss",
        "take_profit",
        "confidence_trend",
        "win_rate",
        "threshold_buy",
        "threshold_sell",
    ]
    available = [c for c in base_cols + extra_cols if c in df.columns]

    display_df = df[available].copy()
    display_df["timeframe"] = display_df["timeframe"].map(TIMEFRAME_LABELS)
    display_df["信心度"] = display_df["confidence"].apply(lambda x: f"{x:.2%}")
    display_df = display_df.drop(columns=["confidence"])

    if "f1_score" in display_df.columns:
        display_df["F1 分數"] = display_df["f1_score"].apply(
            lambda x: f"{x:.4f}" if pd.notna(x) else "-"
        )
    if "auc_score" in display_df.columns:
        display_df["AUC 分數"] = display_df["auc_score"].apply(
            lambda x: f"{x:.4f}" if pd.notna(x) else "-"
        )
    if "expected_return" in display_df.columns:
        display_df["預期報酬"] = display_df["expected_return"].apply(
            lambda x: f"{x:+.2f}%" if pd.notna(x) else "-"
        )
    if "risk_reward" in display_df.columns:
        display_df["風險報酬比"] = display_df["risk_reward"].apply(
            lambda x: f"{x:.2f}" if pd.notna(x) and x > 0 else "-"
        )
    if "stop_loss" in display_df.columns:
        display_df["止損"] = display_df["stop_loss"].apply(
            lambda x: f"{x:+.2f}%" if pd.notna(x) else "-"
        )
    if "take_profit" in display_df.columns:
        display_df["止盈"] = display_df["take_profit"].apply(
            lambda x: f"{x:+.2f}%" if pd.notna(x) else "-"
        )
    if "confidence_trend" in display_df.columns:
        display_df["趨勢"] = display_df["confidence_trend"]
    if "win_rate" in display_df.columns:
        display_df["勝率"] = display_df["win_rate"].apply(
            lambda x: f"{x:.1f}%" if pd.notna(x) else "-"
        )
    if "threshold_buy" in display_df.columns:
        display_df["Buy 閾值"] = display_df["threshold_buy"].apply(
            lambda x: f"{x:.0%}" if pd.notna(x) else "-"
        )
    if "threshold_sell" in display_df.columns:
        display_df["Sell 閾值"] = display_df["threshold_sell"].apply(
            lambda x: f"{x:.0%}" if pd.notna(x) else "-"
        )
    if "created_at" in display_df.columns:
        display_df["預測時間"] = pd.to_datetime(display_df["created_at"]).dt.strftime(
            "%Y-%m-%d %H:%M"
        )

    # Rename
    rename_map = {
        "stock_code": "股票代碼",
        "prediction_date": "預測日期",
        "timeframe": "時間範圍",
        "signal": "信號",
        "model_version": "模型版本",
        "model_type": "冠軍模型",
    }
    display_df = display_df.rename(columns=rename_map)

    # Final column order
    final_cols = [
        "股票代碼",
        "預測日期",
        "時間範圍",
        "信號",
        "信心度",
        "趨勢",
        "Buy 閾值",
        "Sell 閾值",
        "預期報酬",
        "止損",
        "止盈",
        "風險報酬比",
        "勝率",
        "模型版本",
        "冠軍模型",
        "F1 分數",
        "AUC 分數",
        "預測時間",
    ]
    final_cols = [c for c in final_cols if c in display_df.columns]
    display_df = display_df[final_cols]
    display_df = display_df.sort_values("預測時間", ascending=False)

    st.dataframe(display_df, use_container_width=True, hide_index=True)

    # Export functionality
    st.markdown("---")
    st.subheader("📥 匯出預測數據")

    col1, col2 = st.columns(2)

    with col1:
        # Export to CSV
        csv_data = display_df.to_csv(index=False, encoding="utf-8-sig")
        st.download_button(
            label="📄 匯出 CSV",
            data=csv_data,
            file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            help="下載 CSV 格式，可用 Excel 或 Google Sheets 開啟",
        )

    with col2:
        # Export to Excel
        try:
            import openpyxl
            from io import BytesIO

            buffer = BytesIO()
            with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
                display_df.to_excel(writer, index=False, sheet_name="預測記錄")

            st.download_button(
                label="📊 匯出 Excel",
                data=buffer.getvalue(),
                file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                help="下載 Excel 格式，包含格式化的工作表",
            )
        except ImportError:
            st.info("安裝 openpyxl 以啟用 Excel 匯出: `pip install openpyxl`")

    # Model metrics explanation
    if "model_type" in df.columns:
        with st.expander("📖 模型指標說明"):
            st.markdown("""
            | 指標 | 說明 | 範圍 |
            |---|---|---|
            | **F1 分數** | 精準率與召回率的調和平均數。越高表示模型預測越準確（兼顧「預測對的」和「不漏掉」）。0.5 為隨機水平，>0.6 為可用。 | 0 ~ 1 |
            | **AUC 分數** | 模型區分漲跌的能力。0.5 = 隨機猜測，1.0 = 完美區分。衡量模型對信心度排序的品質。 | 0.5 ~ 1.0 |
            | **冠軍模型** | Optuna 自動調參後，集成模型 (voting/stacking) 或單一模型的類型。 | voting / stacking / xgboost / lightgbm |
            | **預期報酬** | 基於模型信心度和歷史波動率估算的預期報酬率。正數=預期上漲，負數=預期下跌。 | ±XX% |
            | **風險報酬比** | 預期收益與潛在風險的比率。>1 表示收益大於風險，<1 表示風險大於收益。 | 0 ~ X |
            | **止損** | 建議止損點。Buy信號為負數（下跌止損），Sell信號為正數（上漲止損）。 | ±XX% |
            | **止盈** | 建議止盈點。Buy信號為正數（上漲獲利），Sell信號為負數（下跌獲利）。 | ±XX% |
            | **趨勢** | 信心度變化趨勢。↑=上升，↓=下降，→=持平，-=首次預測。 | ↑↓→- |
            | **勝率** | 歷史預測準確率（簡化計算： Buy+Sell信號比例）。 | 0~100% |

            **注意：** Buy/Sell 信號閾值由模型自動優化，不再使用固定 55%/45%。每個時間範圍有獨立的最佳閾值。
            """)

            st.markdown("""
            **預期報酬計算公式：**
            ```
            預期報酬 = (信心度 - 0.5) × 2 × 歷史波動率 × √天數 × 100%
            ```

            **風險報酬比計算：**
            - 風險 = 1個標準差的波動（歷史波動率 × √天數）
            - 報酬 = |預期報酬|
            - 風險報酬比 = 報酬 / 風險

            **解讀：**
            - 風險報酬比 > 1：潛在收益大於風險（有利）
            - 風險報酬比 < 1：潛在收益小於風險（不利）
            - 風險報酬比 = 1：收益與風險平衡

            **信號與預期報酬：**
            - **Buy 信號**：預期報酬為正 → 預期上漲，買入獲利
            - **Sell 信號**：預期報酬為負 → 預期下跌，放空獲利（做空）
            - **Hold 信號**：預期報酬接近0 → 無明確方向，觀望
            """)

    # Features explanation
    with st.expander("🔬 模型學習的技術指標 (Features)"):
        st.markdown("""
        模型使用 **3 年歷史數據** (約 750 交易日) 訓練，從 **OHLCV** + **市場指數** 計算以下 33 項特徵：

        | 類別 | 特徵名稱 | 說明 |
        |---|---|---|
        | **報酬率** | `ret_1d`, `ret_3d`, `ret_5d`, `ret_10d`, `ret_20d`, `ret_30d` | 1/3/5/10/20/30日漲跌幅 |
        | **價格形態** | `high_low_range`, `close_to_high`, `close_to_low` | 日內振幅、收盤位置 |
        | **價格位置** | `ma50_deviation` | 當前價格與 50 日均線乖離率 |
        | **成交量** | `vol_ratio_5d`, `vol_ratio_10d` | 量能相對強弱 |
        | **成交量** | `obv_change` | OBV (能量潮) 變化 |
        | **成交量** | `volume_cv` | 成交量變異係數 (20日) |
        | **動量** | `rsi_14` | RSI 超買/超賣 |
        | **動量** | `stoch_k`, `stoch_d` | 隨機震盪指標 |
        | **動量** | `mfi` | 資金流量指標 |
        | **動量** | `williams_r` | 威廉指標 (%R) |
        | **趨勢** | `macd_diff`, `macd_dea`, `macd_hist` | MACD 三元件 |
        | **趨勢** | `adx` | 趨勢強度 (不分方向) |
        | **波動** | `bb_width` | 布林通道寬度 |
        | **波動** | `atr_14`, `atr_ratio` | 平均真實波幅、ATR/收盤價比值 |
        | **統計** | `ret_5d_skew`, `ret_5d_kurt` | 報酬率偏度/峰度 |
        | **統計** | `volatility_10d`, `volatility_20d` | 10日/20日波動率 |
        | **市場** | `hsi_ret_5d`, `hsi_ret_20d` | 恒生指數漲跌幅 |
        | **匯率** | `usdhkd_change` | 美元/港幣匯率變化 |

        **目標變數 (Target)：**
        | 時間範圍 | 說明 |
        |---|---|
        | `1d` | 明日收盤 > 今日收盤 → 1 (上漲), 否則 → 0 |
        | `5d` | 5日後收盤 > 今日收盤 → 1, 否則 → 0 |
        | `20d` | 20日後收盤 > 今日收盤 → 1, 否則 → 0 |

        **模型架構：**
        | 模式 | 說明 |
        |---|---|
        | **Voting** | XGBoost + LightGBM + RandomForest，加權平均預測機率 |
        | **Stacking** | 同上三個基礎模型 + LogisticRegression 元模型 (USE_STACKING=True 時自動強制啟用集成) |
        | **SMOTE** | 訓練折上自動生成少數類合成樣本 (可與任何模式組合) |

        **模型表現 (F1 Score)：**
        | 時間範圍 | F1 Score | 說明 |
        |---|---|---|
        | 1天 | ~0.57 | 可用 — 短期趨勢 |
        | 5天 | ~0.69 | 良好 — 中期動量 |
        | 20天 | ~0.73 | 最佳 — 長期趨勢 |
        """)

with tab_performance:
    # --- Model Monitoring Section ---
    st.markdown("---")
    st.subheader("🔍 模型監控")
    st.caption("自動監控模型性能、數據品質和信號品質，確保預測可靠性")

    # Import monitoring functions
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
    from model_monitoring import (
        DataQualityChecker,
        ModelDriftDetector,
        AlertManager,
        ConfidenceCalibrator,
    )

    # Initialize monitoring
    checker = DataQualityChecker(client)
    drift_detector = ModelDriftDetector(client)
    alert_manager = AlertManager(client)
    calibrator = ConfidenceCalibrator(client)

    # Get stock codes from config
    STOCK_LIST = os.getenv("STOCK_LIST", "0700,9988,0005,0939").split(",")

    # Data Quality Checks
    with st.expander("📊 數據品質檢查"):
        st.info("""
        **檢查項目：**
        - **數據完整性**: 每個時間範圍(1d/5d/20d)至少需要5筆預測記錄
        - **信心度分佈**: 檢查是否有極端值 (>90% 或 <10%) 過多
        - **信號平衡性**: 檢查買入/賣出/持有信號是否過度集中 (>70%)

        **計算方式：**
        - 統計過去30天內每個時間範圍的預測數量
        - 計算信心度分佈，檢測極端值比例
        - 分析信號分佈，確保各信號比例合理
        """)
        quality_results = checker.run_all_checks(STOCK_LIST)
        for result in quality_results:
            # Check both missing_dates and confidence_dist status
            missing_ok = result["missing_dates"]["status"] == "ok"
            dist_ok = result["confidence_dist"]["status"] == "ok"
            status = "✅" if missing_ok and dist_ok else "⚠️"

            st.write(f"{status} **{result['stock_code']}**")

            # Show missing dates status
            st.write(f"  - 數據完整性: {result['missing_dates']['message']}")
            if "issues" in result["missing_dates"]:
                for issue in result["missing_dates"]["issues"]:
                    st.write(f"    - {issue}")

            # Show confidence distribution status
            st.write(f"  - 信心度分佈: {result['confidence_dist']['message']}")
            if "issues" in result["confidence_dist"]:
                for issue in result["confidence_dist"]["issues"]:
                    st.write(f"    - {issue}")

    # Model Drift Detection
    with st.expander("📉 模型漂移檢測"):
        st.info("""
        **檢測原理：**
        比較最近7天與30天的平均信心度，判斷模型是否退化

        **警報等級：**
        - 🔴 **高度警報**: 準確度下降 >10% (模型嚴重退化，需重新訓練)
        - 🟡 **中度警報**: 準確度下降 >5% (模型可能退化，建議重新訓練)
        - ✅ **穩定**: 準確度變化 <5% (模型正常運作)

        **計算公式：**
        ```
        漂移值 = 最近7天平均信心度 - 最近30天平均信心度
        ```
        """)
        drift_results = drift_detector.check_all_models(STOCK_LIST)
        for result in drift_results:
            if result.get("drift"):
                severity = "🔴" if result.get("severity") == "high" else "🟡"
                st.write(
                    f"{severity} **{result['stock_code']}** ({result['timeframe']}): {result['message']}"
                )
            else:
                st.write(
                    f"✅ **{result['stock_code']}** ({result['timeframe']}): {result['message']}"
                )

    # Signal Alerts
    with st.expander("🔔 信號警報"):
        st.info("""
        **警報條件：**
        - **強勢信號**: 信心度 >70% 的買入或賣出信號
        - **高回報**: 預期報酬 >5% 的投資機會

        **如何解讀：**
        - 📈 **買入警報**: 模型強烈預期上漲，可考慮買入
        - 📉 **賣出警報**: 模型強烈預期下跌，可考慮賣出或放空

        **注意：** 信號僅供參考，請結合其他分析判斷
        """)
        alerts = alert_manager.check_alerts(STOCK_LIST)
        if alerts:
            st.markdown(alert_manager.format_alerts(alerts))
        else:
            st.info("目前沒有需要關注的信號。")

    # Confidence Calibration
    with st.expander("🎯 信心度校準"):
        st.info("""
        **校準原理：**
        確保模型輸出的信心度分數可靠，反映真實的預測概率

        **判斷標準：**
        - **過度自信**: 平均信心度 >60% (模型可能高估預測能力)
        - **信心不足**: 平均信心度 <40% (模型可能低估預測能力)
        - **正常範圍**: 平均信心度 40%-60% (信心度可靠)

        **計算方式：**
        - 計算每個股票最近100筆預測的平均信心度
        - 計算標準差 (信心度穩定性)
        - 根據平均值判斷是否需要調整
        """)
        calibration_data = []
        for code in STOCK_LIST:
            calibration = calibrator.calculate_calibration(code)
            if calibration.get("avg_confidence") is not None:
                adjustment = calibrator.suggest_calibration_adjustment(calibration)
                calibration_data.append(
                    {
                        "stock_code": code,
                        "avg_confidence": calibration["avg_confidence"],
                        "std_confidence": calibration.get("std_confidence", 0),
                        "sample_size": calibration.get("sample_size", 0),
                        "message": adjustment["message"],
                    }
                )

        if calibration_data:
            for data in calibration_data:
                st.write(
                    f"**{data['stock_code']}**: 平均信心度 {data['avg_confidence']:.1%} (標準差 {data['std_confidence']:.1%}) - {data['message']}"
                )
                st.write(f"  - 數據量: {data['sample_size']} 筆預測")
        else:
            st.info("數據不足，無法進行信心度校準分析。")

with tab_performance:
    # --- Model Performance Tracking ---
    st.markdown("---")
    st.subheader("📊 模型表現追蹤")
    st.caption("追蹤模型預測的真實準確度，而非僅看訓練指標")

    # Rolling accuracy section
    with st.expander("📈 滾動準確度 (Rolling Accuracy)", expanded=True):
        st.info("""
        **計算方式：**
        對每個預測，驗證 N 天後的實際價格是否朝預測方向移動：
        - Buy 信號 → N天後收盤價 > 預測日收盤價 = 正確
        - Sell 信號 → N天後收盤價 < 預測日收盤價 = 正確
        - Hold 信號 → 不計入

        **滾動 30 天準確度** = 最近 30 天內正確預測數 / 總預測數 × 100%
        """)

        perf_stock = st.selectbox(
            "選擇股票",
            options=stock_codes_in_df,
            key="perf_stock",
        )
        perf_tf = st.selectbox(
            "選擇時間範圍",
            options=["1d", "5d", "20d"],
            format_func=lambda x: TIMEFRAME_LABELS[x],
            key="perf_tf",
        )

        if perf_stock:
            # Calculate rolling accuracy using ModelDriftDetector
            from model_monitoring import ModelDriftDetector
            drift_detector_perf = ModelDriftDetector(client)

            # Get accuracy data
            accuracy_result = drift_detector_perf.calculate_accuracy(
                perf_stock, perf_tf, days=30
            )

            if accuracy_result.get("accuracy") is not None:
                acc = accuracy_result["accuracy"]
                total = accuracy_result.get("total", 0)
                correct = accuracy_result.get("correct", 0)

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(
                        "30天滾動準確度",
                        f"{acc:.1f}%",
                        help="最近30天內正確預測的比例",
                    )
                with col2:
                    st.metric("正確預測數", f"{correct}")
                with col3:
                    st.metric("總預測數", f"{total}")

                # Show accuracy breakdown by signal type
                if "details" in accuracy_result:
                    details = accuracy_result["details"]
                    if details:
                        st.markdown("**各信號類型表現：**")
                        detail_df = pd.DataFrame(details)
                        if not detail_df.empty:
                            st.dataframe(detail_df, use_container_width=True, hide_index=True)
            else:
                st.info("數據不足，無法計算滾動準確度。需至少有已過期的預測記錄。")

    # Training metrics (static F1/AUC from DB)
    with st.expander("📋 訓練指標 (F1 / AUC)"):
        st.info("""
        **說明：** 以下指標為模型訓練時計算的 F1 Score 和 AUC Score。
        這些是靜態指標，每次重新訓練後更新。

        - **F1 Score**: 精準率與召回率的調和平均數 (>0.5 可用, >0.6 良好)
        - **AUC Score**: 模型區分漲跌的能力 (0.5=隨機, >0.6 可用, >0.7 良好)
        """)

        # Get latest training metrics from predictions
        if not df.empty and "f1_score" in df.columns:
            # Group by stock_code and timeframe, get latest metrics
            metrics_data = []
            for stock in stock_codes_in_df:
                for tf in ["1d", "5d", "20d"]:
                    tf_label = TIMEFRAME_LABELS.get(tf, tf)
                    subset = df[
                        (df["stock_code"] == stock)
                        & (df["timeframe"] == tf)
                        & (df["f1_score"].notna())
                    ]
                    if not subset.empty:
                        latest = subset.sort_values("prediction_date").iloc[-1]
                        metrics_data.append({
                            "股票": stock,
                            "時間範圍": tf_label,
                            "F1 Score": latest.get("f1_score"),
                            "AUC Score": latest.get("auc_score"),
                            "冠軍模型": latest.get("model_type", "-"),
                            "更新日期": latest.get("prediction_date", "-"),
                        })

            if metrics_data:
                metrics_df = pd.DataFrame(metrics_data)
                # Format scores
                metrics_df["F1 Score"] = metrics_df["F1 Score"].apply(
                    lambda x: f"{x:.4f}" if pd.notna(x) else "-"
                )
                metrics_df["AUC Score"] = metrics_df["AUC Score"].apply(
                    lambda x: f"{x:.4f}" if pd.notna(x) else "-"
                )
                st.dataframe(metrics_df, use_container_width=True, hide_index=True)
            else:
                st.info("尚無訓練指標數據。請先執行 `python src/train_model.py` 訓練模型。")
        else:
            st.info("尚無訓練指標數據。請先執行 `python src/train_model.py` 訓練模型。")


# --- Footer ---
st.markdown("---")
st.caption("⚠️ 本系統僅供參考，不構成投資建議。投資有風險，入市需謹慎。")
