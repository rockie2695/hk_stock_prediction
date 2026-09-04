"""
Investment Simulation Dashboard - Simulate following Buy/Sell signals.
Run with: streamlit run app/streamlit_app.py (appears in sidebar nav)
"""
import os
import sys
from datetime import datetime, timedelta
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Ensure project root is importable
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from config import STOCK_LIST
from src.simulator import simulate_investment, SimulationResult

# Page config
st.set_page_config(
    page_title="💰 投資模擬器",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("💰 投資模擬器")
st.caption("根據預測信號模擬投資，計算實際收益與風險")

# --- Sidebar Controls ---
st.sidebar.header("⚙️ 模擬設定")

col1, col2 = st.sidebar.columns(2)
with col1:
    start_date = st.date_input(
        "開始日期",
        value=(datetime.now() - timedelta(days=90)).date(),
        key="sim_start",
    )
with col2:
    end_date = st.date_input(
        "結束日期",
        value=datetime.now().date(),
        key="sim_end",
    )

initial_capital = st.sidebar.number_input(
    "初始資金 (HKD)",
    min_value=1000,
    max_value=1000000,
    value=20000,
    step=1000,
    key="sim_capital",
)

timeframe = st.sidebar.selectbox(
    "時間範圍",
    options=["1d", "5d", "20d"],
    format_func=lambda x: {"1d": "明日 (1天)", "5d": "下週 (5天)", "20d": "下月 (20天)"}[x],
    key="sim_tf",
)

selected_stocks = st.sidebar.multiselect(
    "選擇股票",
    options=STOCK_LIST,
    default=STOCK_LIST,
    key="sim_stocks",
)

optimize_timing = st.sidebar.checkbox(
    "🎯 最佳時機模式",
    value=False,
    help="假設預知未來N天價格，選最佳買賣日（僅5d/20d有效）",
    key="sim_optimize",
)

run_simulation = st.sidebar.button("🚀 開始模擬", type="primary", use_container_width=True)

# --- Run Simulation ---
if run_simulation:
    if not selected_stocks:
        st.warning("請至少選擇一檔股票")
        st.stop()

    if start_date >= end_date:
        st.warning("開始日期必須早於結束日期")
        st.stop()

    st.markdown("---")

    with st.spinner("🔄 模擬中...正在獲取預測數據並模擬交易"):
        results = {}
        opt_results = {}
        progress = st.progress(0)
        for i, code in enumerate(selected_stocks):
            # Normal mode
            result = simulate_investment(
                stock_code=code,
                start_date=start_date.isoformat(),
                end_date=end_date.isoformat(),
                initial_capital=initial_capital,
                timeframe=timeframe,
                optimize_timing=False,
            )
            if result is not None:
                results[code] = result

            # Optimized mode (if enabled)
            if optimize_timing:
                opt_result = simulate_investment(
                    stock_code=code,
                    start_date=start_date.isoformat(),
                    end_date=end_date.isoformat(),
                    initial_capital=initial_capital,
                    timeframe=timeframe,
                    optimize_timing=True,
                )
                if opt_result is not None:
                    opt_results[code] = opt_result

            progress.progress((i + 1) / len(selected_stocks))
        progress.empty()

    if not results:
        st.error("無法獲取模擬數據。請確認日期範圍內有預測記錄。")
        st.stop()

    # Store in session state for display
    st.session_state['sim_results'] = results
    st.session_state['sim_opt_results'] = opt_results
    st.session_state['sim_params'] = {
        'start_date': start_date,
        'end_date': end_date,
        'initial_capital': initial_capital,
        'timeframe': timeframe,
        'optimize_timing': optimize_timing,
    }

# --- Display Results ---
if 'sim_results' in st.session_state and st.session_state['sim_results']:
    results = st.session_state['sim_results']
    opt_results = st.session_state.get('sim_opt_results', {})
    params = st.session_state['sim_params']
    show_comparison = params.get('optimize_timing', False) and opt_results

    def render_results(res, title, show_detail=True):
        """Render results for a given set of simulation results."""
        total_initial = params['initial_capital'] * len(res)
        total_final = sum(r.final_value for r in res.values())
        total_pnl = total_final - total_initial
        total_return = ((total_final - total_initial) / total_initial) * 100 if total_initial > 0 else 0
        total_trades = sum(r.total_trades for r in res.values())
        total_wins = sum(r.wins for r in res.values())
        total_sells = sum(r.sell_trades for r in res.values())
        overall_win_rate = (total_wins / total_sells * 100) if total_sells > 0 else 0
        max_dd = max((r.max_drawdown_pct for r in res.values()), default=0)

        # Summary cards
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            color = "normal" if total_pnl >= 0 else "inverse"
            st.metric(
                "總盈虧",
                f"HKD {total_pnl:+,.0f}",
                delta=f"{total_return:+.1f}%",
                delta_color=color,
            )
        with c2:
            st.metric("總交易次數", f"{total_trades}")
        with c3:
            st.metric("勝率", f"{overall_win_rate:.1f}%")
        with c4:
            st.metric("最大回撤", f"{max_dd:.1f}%",
                       help="從最高點到最低點的最大跌幅百分比。例如：投資組合從 HKD 20,000 跌到 HKD 18,000，回撤 = 10%")
        with c5:
            st.metric("總交易成本", f"HKD {sum(r.total_commission + r.total_stamp_duty for r in res.values()):,.0f}")

        # Per-stock breakdown
        if show_detail:
            st.markdown("##### 📋 各股票表現")
            rows = []
            for code, r in res.items():
                pnl = r.final_value - r.initial_capital
                rows.append({
                    "股票代碼": code,
                    "初始資金": f"HKD {r.initial_capital:,.0f}",
                    "最終價值": f"HKD {r.final_value:,.0f}",
                    "盈虧": f"HKD {pnl:+,.0f}",
                    "報酬率": f"{r.total_return_pct:+.1f}%",
                    "買入次數": r.buy_trades,
                    "賣出次數": r.sell_trades,
                    "勝率": f"{r.win_rate:.1f}%",
                    "最大回撤": f"{r.max_drawdown_pct:.1f}%",
                    "交易成本": f"HKD {r.total_commission + r.total_stamp_duty:,.0f}",
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # Trade history
        all_trades = []
        for code, r in res.items():
            for t in r.trade_log:
                all_trades.append({
                    "日期": t.date,
                    "股票代碼": t.stock_code,
                    "操作": t.action,
                    "價格": f"{t.price:.2f}",
                    "股數": t.shares,
                    "交易成本": f"HKD {t.cost:.2f}",
                    "盈虧": f"HKD {t.pnl:+,.2f}" if t.action == "Sell" else "-",
                })
        if all_trades:
            trades_df = pd.DataFrame(all_trades).sort_values("日期", ascending=False)
            st.dataframe(trades_df, use_container_width=True, hide_index=True)
        else:
            st.info("模擬期間內無交易記錄。")

    def render_charts(res, mode_label):
        """Render portfolio value charts for given results."""
        st.markdown("---")
        st.subheader("📈 投資組合價值走勢")

        chart_data = []
        for code, r in res.items():
            for snap in r.portfolio_history:
                chart_data.append({
                    "日期": snap.date,
                    "股票代碼": code,
                    "投資組合價值": snap.portfolio_value,
                    "現金": snap.cash,
                    "持股價值": snap.stock_value,
                })

        if not chart_data:
            st.info("無圖表數據。")
            return

        chart_df = pd.DataFrame(chart_data)

        fig = px.line(
            chart_df,
            x="日期",
            y="投資組合價值",
            color="股票代碼",
            labels={
                "日期": "日期",
                "投資組合價值": "價值 (HKD)",
                "股票代碼": "股票",
            },
            title=f"各股票投資組合價值 ({params['start_date']} ~ {params['end_date']})",
        )

        fig.add_hline(
            y=params['initial_capital'],
            line_dash="dash",
            line_color="gray",
            annotation_text=f"初始資金 HKD {params['initial_capital']:,}",
        )

        fig.update_layout(yaxis_tickformat=",.0f")
        st.plotly_chart(fig, use_container_width=True, key=f"chart_line_{mode_label}")

        # Stacked area chart for total portfolio
        total_chart_data = []
        for snap_date in chart_df['日期'].unique():
            day_data = chart_df[chart_df['日期'] == snap_date]
            total_value = day_data['投資組合價值'].sum()
            total_chart_data.append({
                "日期": snap_date,
                "總投資組合價值": total_value,
            })

        if total_chart_data:
            total_df = pd.DataFrame(total_chart_data)
            fig_total = px.area(
                total_df,
                x="日期",
                y="總投資組合價值",
                labels={"日期": "日期", "總投資組合價值": "總價值 (HKD)"},
                title="投資組合總價值走勢",
            )
            total_initial = params['initial_capital'] * len(res)
            fig_total.add_hline(
                y=total_initial,
                line_dash="dash",
                line_color="gray",
                annotation_text=f"總初始資金 HKD {total_initial:,}",
            )
            fig_total.update_layout(yaxis_tickformat=",.0f")
            st.plotly_chart(fig_total, use_container_width=True, key=f"chart_area_{mode_label}")

    def render_trades(res, mode_label):
        """Render trade history table for given results."""
        st.markdown("---")
        st.subheader("📝 交易記錄")

        all_trades = []
        for code, r in res.items():
            for t in r.trade_log:
                all_trades.append({
                    "日期": t.date,
                    "股票代碼": t.stock_code,
                    "操作": t.action,
                    "價格": f"{t.price:.2f}",
                    "股數": t.shares,
                    "交易成本": f"HKD {t.cost:.2f}",
                    "盈虧": f"HKD {t.pnl:+,.2f}" if t.action == "Sell" else "-",
                })

        if all_trades:
            trades_df = pd.DataFrame(all_trades)
            trades_df = trades_df.sort_values("日期", ascending=False)
            st.dataframe(trades_df, use_container_width=True, hide_index=True)

            csv = trades_df.to_csv(index=False, encoding="utf-8-sig")
            st.download_button(
                label="📥 匯出交易記錄 (CSV)",
                data=csv,
                file_name=f"trades_{mode_label}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
            )
        else:
            st.info("模擬期間內無交易記錄。")

    def render_details(res, mode_label):
        """Render per-stock detail expanders for given results."""
        st.markdown("---")
        st.subheader("🔍 各股票詳細分析")

        for code, r in res.items():
            with st.expander(f"📊 {code} 詳細分析"):
                pnl = r.final_value - r.initial_capital

                dc1, dc2, dc3, dc4 = st.columns(4)
                with dc1:
                    st.metric("最終價值", f"HKD {r.final_value:,.0f}")
                with dc2:
                    st.metric("盈虧", f"HKD {pnl:+,.0f}", delta=f"{r.total_return_pct:+.1f}%")
                with dc3:
                    st.metric("交易次數", f"{r.total_trades} (買{r.buy_trades}/賣{r.sell_trades})")
                with dc4:
                    st.metric("最大回撤", f"{r.max_drawdown_pct:.1f}%",
                               help="從最高點到最低點的最大跌幅百分比。例如：投資組合從 HKD 20,000 跌到 HKD 18,000，回撤 = 10%")

                if r.portfolio_history:
                    snap_df = pd.DataFrame([{
                        "日期": s.date,
                        "現金": s.cash,
                        "持股價值": s.stock_value,
                        "總價值": s.portfolio_value,
                    } for s in r.portfolio_history])

                    fig_stock = go.Figure()
                    fig_stock.add_trace(go.Scatter(
                        x=snap_df['日期'], y=snap_df['總價值'],
                        mode='lines', name='總價值',
                        line=dict(color='#2ecc71', width=2),
                    ))
                    fig_stock.add_trace(go.Scatter(
                        x=snap_df['日期'], y=snap_df['現金'],
                        mode='lines', name='現金',
                        line=dict(color='#3498db', width=1, dash='dash'),
                    ))
                    fig_stock.add_trace(go.Scatter(
                        x=snap_df['日期'], y=snap_df['持股價值'],
                        mode='lines', name='持股價值',
                        line=dict(color='#e74c3c', width=1, dash='dot'),
                    ))
                    fig_stock.update_layout(
                        title=f"{code} 投資組合構成",
                        yaxis_title="價值 (HKD)",
                        yaxis_tickformat=",.0f",
                        hovermode="x unified",
                    )
                    st.plotly_chart(fig_stock, use_container_width=True, key=f"detail_{mode_label}_{code}")

                if r.trade_log:
                    stock_trades = pd.DataFrame([{
                        "日期": t.date,
                        "操作": t.action,
                        "價格": f"{t.price:.2f}",
                        "股數": t.shares,
                        "成本": f"HKD {t.cost:.2f}",
                        "盈虧": f"HKD {t.pnl:+,.2f}" if t.action == "Sell" else "-",
                    } for t in r.trade_log])
                    stock_trades = stock_trades.sort_values("日期", ascending=False)
                    st.dataframe(stock_trades, use_container_width=True, hide_index=True)

    # --- Summary Cards ---
    st.markdown("---")
    st.subheader("📊 模擬總覽")

    if show_comparison:
        tab_normal, tab_opt = st.tabs(["📊 一般模式", "🎯 最佳時機模式"])
        with tab_normal:
            render_results(results, "一般模式", show_detail=True)
            render_charts(results, "normal")
            render_trades(results, "一般")
            render_details(results, "normal")
        with tab_opt:
            render_results(opt_results, "最佳時機模式", show_detail=True)
            render_charts(opt_results, "optimal")
            render_trades(opt_results, "最佳時機")
            render_details(opt_results, "optimal")
    else:
        render_results(results, "一般模式", show_detail=True)
        render_charts(results, "normal")
        render_trades(results, "一般")
        render_details(results, "normal")

else:
    st.info("👈 設定參數後點擊「開始模擬」查看結果")

    # Show explanation
    with st.expander("📖 模擬規則說明"):
        st.markdown("""
        ### 模擬規則

        | 信號 | 操作 | 說明 |
        |------|------|------|
        | **Buy** | 買入 | 以當日收盤價買入最大可購入股數 (整股) |
        | **Sell** | 賣出 | 以當日收盤價賣出所有持股 |
        | **Hold** | 觀望 | 不進行任何交易 |

        ### 最佳時機模式 (Hindsight)

        啟用後，系統會在 N 天窗口內尋找最佳買賣日：

        | 時間範圍 | Buy 策略 | Sell 策略 |
        |----------|----------|-----------|
        | **5d** | 在信號日後 5 天內找**最低價**日買入 | 在信號日後 5 天內找**最高價**日賣出 |
        | **20d** | 在信號日後 20 天內找**最低價**日買入 | 在信號日後 20 天內找**最高價**日賣出 |
        | **1d** | 當日收盤價買入 (不適用) | 當日收盤價賣出 (不適用) |

        > ⚠️ 這是**後見之明**模式，實際交易無法預知未來價格。用於評估信號的最佳潛在表現。

        ### 交易成本

        | 費用 | 比例 | 說明 |
        |------|------|------|
        | **佣金** | 0.1% | 每筆交易最低 HKD 20 |
        | **印花稅** | 0.13% | 僅賣出時收取 |

        ### 計算方式

        - **買入股數** = 可投資金額 ÷ (股價 × (1 + 佣金率))，取整數
        - **盈虧** = 賣出所得 - 買入成本 - 所有交易費用
        - **最大回撤** = 歷史最高點到最低點的跌幅百分比 (例：從 HKD 20,000 跌到 HKD 18,000 = 10%)
        - **勝率** = 盈利交易次數 ÷ 總賣出次數 × 100%

        ### 注意事項

        - 本模擬僅供參考，不構成投資建議
        - 實際交易可能有滑點、流動性等影響
        - 不考慮做空（Sell 信號僅用於平倉）
        """)

# --- Footer ---
st.markdown("---")
st.caption("⚠️ 本模擬僅供參考，不構成投資建議。投資有風險，入市需謹慎。")
