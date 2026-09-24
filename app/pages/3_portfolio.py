"""
Portfolio View / 投資組合視圖
Current holdings P&L, allocation, and risk exposure.
當前持倉盈虧、配置和風險暴露。

Run with / 執行方式: streamlit run app/streamlit_app.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import pytz
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.data_fetcher import fetch_stock_data
from src.logger import setup_logger

logger = setup_logger("portfolio_page")

HK_TZ = pytz.timezone("Asia/Hong_Kong")
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@st.cache_data(ttl=300)
def get_current_signals(stock_codes):
    """Get latest predictions from Supabase / 從 Supabase 獲取最新預測 (cached 5 min)"""
    try:
        from supabase import create_client
        from config import SUPABASE_URL, SUPABASE_KEY
        client = create_client(SUPABASE_URL, SUPABASE_KEY)
        records = []
        for code in stock_codes:
            result = client.table("stock_predictions").select("*").eq(
                "stock_code", code
            ).order("created_at", desc=True).limit(3).execute()
            if result.data:
                for r in result.data:
                    records.append(r)
        return pd.DataFrame(records) if records else pd.DataFrame()
    except Exception as e:
        logger.warning(f"Failed to fetch signals: {e} / 無法獲取信號")
        return pd.DataFrame()


def render_portfolio_page():
    """Render the portfolio page / 渲染投資組合頁面"""
    st.header("Portfolio / 投資組合")
    st.caption("Current holdings, P&L, and risk exposure / 當前持倉、盈虧和風險暴露")
    
    from config import STOCK_LIST
    predictions_df = get_current_signals(tuple(STOCK_LIST))
    
    if predictions_df.empty:
        st.info("No prediction data available / 暫無預測數據")
        return
    
    # Get latest prediction per stock / 取得每支股票的最新預測
    latest_preds = predictions_df.sort_values("prediction_date").groupby("stock_code").last().reset_index()
    
    # Holdings summary / 持倉總覽
    st.subheader("Holdings Summary / 持倉總覽")
    
    holdings_data = []
    for _, row in latest_preds.iterrows():
        stock = row["stock_code"]
        signal = row["signal"]
        confidence = row["confidence"]
        try:
            df = fetch_stock_data(stock, years=1)
            current_price = float(df["Close"].iloc[-1])
            prev_price = float(df["Close"].iloc[-2]) if len(df) > 1 else current_price
            daily_change = (current_price - prev_price) / prev_price * 100
        except Exception:
            current_price = 0
            daily_change = 0
        holdings_data.append({
            "Stock / 股票": stock,
            "Signal / 信號": signal,
            "Confidence / 信心度": f"{confidence:.1%}",
            "Price / 價格": f"HKD {current_price:.2f}",
            "Daily Change / 日變動": f"{daily_change:+.2f}%"
        })
    
    holdings_df = pd.DataFrame(holdings_data)
    st.dataframe(holdings_df, use_container_width=True, hide_index=True)
    
    # Signal and confidence distribution / 信號和信心度分佈
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Signal Distribution / 信號分佈")
        signal_counts = latest_preds["signal"].value_counts()
        fig = px.pie(values=signal_counts.values, names=signal_counts.index,
                     color_discrete_map={"Buy": "green", "Sell": "red", "Hold": "gray"})
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Confidence Distribution / 信心度分佈")
        fig = px.histogram(latest_preds, x="confidence", nbins=20, color="signal",
                          color_discrete_map={"Buy": "green", "Sell": "red", "Hold": "gray"})
        fig.update_layout(height=300, xaxis_title="Confidence / 信心度", yaxis_title="Count / 數量")
        st.plotly_chart(fig, use_container_width=True)
    
    # Risk exposure / 風險暴露
    st.subheader("Risk Exposure / 風險暴露")
    total_stocks = len(latest_preds)
    buy_count = len(latest_preds[latest_preds["signal"] == "Buy"])
    sell_count = len(latest_preds[latest_preds["signal"] == "Sell"])
    hold_count = len(latest_preds[latest_preds["signal"] == "Hold"])
    avg_confidence = latest_preds["confidence"].mean()
    
    r1, r2, r3, r4, r5 = st.columns(5)
    r1.metric("Total Stocks / 總持股", str(total_stocks))
    r2.metric("Buy Signals / 買入信號", str(buy_count))
    r3.metric("Sell Signals / 賣出信號", str(sell_count))
    r4.metric("Hold Signals / 持有信號", str(hold_count))
    r5.metric("Avg Confidence / 平均信心度", f"{avg_confidence:.1%}")
    
    # Risk warnings / 風險警告
    if buy_count > total_stocks * 0.7:
        st.warning("High concentration of Buy signals / 買入信號過度集中")
    if avg_confidence < 0.55:
        st.warning("Low average confidence / 平均信心度偏低")


if __name__ == "__main__":
    render_portfolio_page()
