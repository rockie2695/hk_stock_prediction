"""
Backtest Page - 策略回測頁面
In-dashboard strategy backtesting with equity curve, drawdown, trade log.
在儀表板中進行策略回測，包含資金曲線、回撤、交易記錄。

Run with / 執行方式: streamlit run app/streamlit_app.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import pytz
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.simulator import simulate_investment, simulate_buy_and_hold
from src.data_fetcher import fetch_stock_data
from src.logger import setup_logger

logger = setup_logger("backtest_page")

HK_TZ = pytz.timezone("Asia/Hong_Kong")
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def get_predictions(stock_code, start_date, end_date, timeframe="5d"):
    """Fetch predictions from Supabase / 從 Supabase 獲取預測數據"""
    try:
        from supabase import create_client
        from config import SUPABASE_URL, SUPABASE_KEY
        client = create_client(SUPABASE_URL, SUPABASE_KEY)
        result = client.table("stock_predictions").select("*").eq(
            "stock_code", stock_code
        ).eq(
            "timeframe", timeframe
        ).gte("prediction_date", start_date).lte(
            "prediction_date", end_date
        ).order("prediction_date").execute()
        return pd.DataFrame(result.data) if result.data else pd.DataFrame()
    except Exception as e:
        logger.warning(f"Failed to fetch predictions: {e} / 無法獲取預測數據")
        return pd.DataFrame()


def render_backtest_page():
    """Render the backtest page / 渲染回測頁面"""
    st.header("Strategy Backtest / 策略回測")
    st.caption("Backtest historical prediction signals / 回測歷史預測信號")
    
    # Input controls / 輸入控制項
    col1, col2, col3 = st.columns(3)
    with col1:
        stock_code = st.text_input("Stock Code / 股票代碼", value="0700", key="bt_stock")
    with col2:
        timeframe = st.selectbox("Timeframe / 時間範圍", ["1d", "5d", "20d"], index=1, key="bt_tf")
    with col3:
        capital = st.number_input("Initial Capital (HKD) / 初始資金", min_value=10000, value=100000, step=10000, key="bt_capital")
    
    col4, col5 = st.columns(2)
    with col4:
        start_date = st.date_input("Start Date / 開始日期", value=datetime.now(HK_TZ).date() - timedelta(days=180), key="bt_start")
    with col5:
        end_date = st.date_input("End Date / 結束日期", value=datetime.now(HK_TZ).date(), key="bt_end")
    
    slippage = st.slider("Slippage (%) / 滑點", 0.0, 1.0, 0.1, 0.1, key="bt_slip") / 100
    
    if st.button("Run Backtest / 執行回測", key="bt_run"):
        with st.spinner("Running backtest... / 執行回測中..."):
            result = simulate_investment(
                stock_code=stock_code,
                start_date=start_date.isoformat(),
                end_date=end_date.isoformat(),
                capital=capital,
                timeframe=timeframe,
                optimize_timing=False,
                slippage=slippage,
                use_stop_loss=True
            )
            
            if result is None:
                st.error("Backtest failed / 回測失敗")
                return
            
            # Results summary / 結果摘要
            st.subheader("Results / 回測結果")
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("P/L / 盈虧", f"HKD {result.total_pnl:,.0f}", f"{result.total_return:.1f}%")
            m2.metric("Win Rate / 勝率", f"{result.win_rate:.1f}%")
            m3.metric("Trades / 交易次數", str(result.trade_count))
            m4.metric("Max DD / 最大回撤", f"{result.max_drawdown:.1f}%")
            m5.metric("Sharpe Ratio", f"{result.sharpe_ratio:.2f}")
            
            # Equity curve and drawdown chart / 資金曲線和回撤圖表
            fig = make_subplots(
                rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                subplot_titles=("Portfolio Value / 資金曲線", "Drawdown / 回撤"), row_heights=[0.7, 0.3]
            )
            
            dates = [s.date for s in result.portfolio_history]
            values = [s.portfolio_value for s in result.portfolio_history]
            fig.add_trace(go.Scatter(x=dates, y=values, name="Portfolio / 策略", line=dict(color="blue", width=2)), row=1, col=1)
            
            # Benchmark: Buy & Hold / 基準：買入持有
            bh_result = simulate_buy_and_hold(stock_code=stock_code, start_date=start_date.isoformat(), end_date=end_date.isoformat(), capital=capital)
            if bh_result:
                bh_dates = [s.date for s in bh_result.portfolio_history]
                bh_values = [s.portfolio_value for s in bh_result.portfolio_history]
                fig.add_trace(go.Scatter(x=bh_dates, y=bh_values, name="Buy & Hold / 買入持有", line=dict(color="gray", dash="dash", width=1)), row=1, col=1)
            
            # Drawdown calculation / 回撤計算
            peak = pd.Series(values).cummax()
            drawdown = (pd.Series(values) - peak) / peak * 100
            fig.add_trace(go.Scatter(x=dates, y=drawdown, name="Drawdown / 回撤", fill="tozeroy", line=dict(color="red", width=1)), row=2, col=1)
            
            fig.update_layout(height=600, title=f"{stock_code} Backtest ({timeframe})")
            fig.update_yaxes(title_text="Value (HKD) / 價值", row=1, col=1)
            fig.update_yaxes(title_text="Drawdown (%) / 回撤", row=2, col=1)
            st.plotly_chart(fig, use_container_width=True)
            
            # Trade log / 交易記錄
            if result.trade_log:
                st.subheader("Trade Log / 交易記錄")
                trade_df = pd.DataFrame([{
                    "Date / 日期": t.date,
                    "Action / 操作": t.action,
                    "Price / 價格": f"{t.price:.2f}",
                    "Shares / 股數": t.shares,
                    "Cost / 成本": f"{t.cost:,.0f}",
                    "P / L / 盈虧": f"{t.pnl:,.0f}" if t.pnl != 0 else "-"
                } for t in result.trade_log])
                st.dataframe(trade_df, use_container_width=True, hide_index=True)
            
            # Additional metrics / 額外指標
            col6, col7, col7b, col8 = st.columns(4)
            col6.metric("Sortino Ratio", f"{result.sortino_ratio:.2f}")
            col7.metric("Profit Factor / 利潤因子", f"{result.profit_factor:.2f}")
            col7b.metric("Avg Hold / 平均持倉", f"{result.avg_holding_days:.0f} days")
            col8.metric("Alpha / 超額報酬", f"{result.alpha:.1f}%")
            
            # Benchmark comparison / 基準比較
            if bh_result:
                st.metric("Buy & Hold / 買入持有", f"HKD {bh_result.total_pnl:,.0f}", f"{bh_result.total_return:.1f}%")


if __name__ == "__main__":
    render_backtest_page()
