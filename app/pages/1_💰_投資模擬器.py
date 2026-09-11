"""
Investment Simulation Dashboard - Simulate following Buy/Sell signals.
Run with: streamlit run app/streamlit_app.py (appears in sidebar nav)
"""
import os
import sys
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Ensure project root is importable
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from config import STOCK_LIST, SUPABASE_URL, SUPABASE_KEY
from src.simulator import (
    simulate_investment, simulate_all_stocks, simulate_portfolio,
    simulate_with_confidence_weighting, monte_carlo_test, SimulationResult,
)

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

slippage_pct = st.sidebar.slider(
    "📊 滑點 (Slippage)",
    min_value=0.0,
    max_value=0.01,
    value=0.001,
    step=0.0005,
    format="%.3f",
    help="模擬實際交易滑點。0.1% = 0.001，買入價格略高、賣出價格略低",
    key="sim_slippage",
)

use_stop_loss = st.sidebar.checkbox(
    "🛑 止損/止盈執行",
    value=False,
    help="根據預測的止損/止盈水平自動平倉（在信號日之間檢查每日價格）",
    key="sim_stop_loss",
)

st.sidebar.markdown("---")
st.sidebar.subheader("🔬 進階功能")

show_portfolio = st.sidebar.checkbox(
    "📊 組合模擬",
    value=False,
    help="模擬所有選中股票的組合表現（資金平均分配）",
    key="chk_portfolio",
)

show_confidence = st.sidebar.checkbox(
    "🎯 信心度加權",
    value=False,
    help="根據信號信心度調整倉位大小（高信心=大倉位）",
    key="chk_confidence",
)

show_monte_carlo = st.sidebar.checkbox(
    "🎲 蒙地卡羅測試",
    value=False,
    help="隨機翻轉信號，測試策略穩健性",
    key="chk_monte_carlo",
)

if show_monte_carlo:
    mc_simulations = st.sidebar.slider(
        "模擬次數",
        min_value=100,
        max_value=5000,
        value=1000,
        step=100,
        key="mc_sims",
    )
    mc_flip_prob = st.sidebar.slider(
        "信號翻轉概率",
        min_value=0.05,
        max_value=0.50,
        value=0.20,
        step=0.05,
        key="mc_flip",
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
                slippage_pct=slippage_pct,
                use_stop_loss=use_stop_loss,
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
                    slippage_pct=slippage_pct,
                    use_stop_loss=use_stop_loss,
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

    # Run advanced features if selected
    if show_portfolio and len(selected_stocks) > 1:
        with st.spinner("📊 模擬組合表現..."):
            portfolio_result = simulate_portfolio(
                stock_codes=selected_stocks,
                start_date=start_date.isoformat(),
                end_date=end_date.isoformat(),
                total_capital=initial_capital * len(selected_stocks),
                timeframe=timeframe,
                optimize_timing=optimize_timing,
            )
            if portfolio_result:
                st.session_state['sim_portfolio'] = portfolio_result

    if show_confidence:
        confidence_results = {}
        with st.spinner("🎯 模擬信心度加權策略..."):
            for code in selected_stocks:
                conf_result = simulate_with_confidence_weighting(
                    stock_code=code,
                    start_date=start_date.isoformat(),
                    end_date=end_date.isoformat(),
                    initial_capital=initial_capital,
                    timeframe=timeframe,
                )
                if conf_result is not None:
                    confidence_results[code] = conf_result
        if confidence_results:
            st.session_state['sim_confidence'] = confidence_results

    if show_monte_carlo:
        mc_results = {}
        with st.spinner(f"🎲 執行蒙地卡羅測試 ({mc_simulations}次模擬)..."):
            for code in selected_stocks:
                mc_result = monte_carlo_test(
                    stock_code=code,
                    start_date=start_date.isoformat(),
                    end_date=end_date.isoformat(),
                    initial_capital=initial_capital,
                    timeframe=timeframe,
                    n_simulations=mc_simulations,
                    flip_probability=mc_flip_prob,
                )
                if 'error' not in mc_result:
                    mc_results[code] = mc_result
        if mc_results:
            st.session_state['sim_monte_carlo'] = mc_results

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
        total_losses = sum(r.losses for r in res.values())
        overall_win_rate = (total_wins / (total_wins + total_losses) * 100) if (total_wins + total_losses) > 0 else 0
        max_dd = max((r.max_drawdown_pct for r in res.values()), default=0)
        avg_sharpe = np.mean([r.sharpe_ratio for r in res.values()]) if res else 0
        avg_sortino = np.mean([r.sortino_ratio for r in res.values()]) if res else 0
        avg_pf = np.mean([r.profit_factor for r in res.values() if r.profit_factor < 999]) if res else 0
        avg_hold = np.mean([r.avg_holding_days for r in res.values() if r.avg_holding_days > 0]) if res else 0

        # Summary cards row 1
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

        # Summary cards row 2 (risk metrics)
        st.markdown("##### 📐 風險指標")
        r1, r2, r3, r4 = st.columns(4)
        with r1:
            st.metric("Sharpe Ratio", f"{avg_sharpe:.2f}",
                       help="年化風險調整報酬率。>1 表示不錯，>2 表示很好，<0 表示不如無風險投資")
        with r2:
            st.metric("Sortino Ratio", f"{avg_sortino:.2f}",
                       help="只考慮下跌風險的風險調整報酬率。比 Sharpe 更專注於下行風險")
        with r3:
            st.metric("利潤因子", f"{avg_pf:.2f}",
                       help="總盈利 / 總虧損。>1 表示盈利大於虧損，>2 表示很好")
        with r4:
            st.metric("平均持倉天數", f"{avg_hold:.1f} 天",
                       help="每次買入到賣出的平均天數")

        # Benchmark comparison
        benchmarks = [r.benchmark for r in res.values() if r.benchmark is not None]
        if benchmarks:
            st.markdown("##### 📊 對比基準 (買入持有)")
            b1, b2, b3 = st.columns(3)
            bh_return = np.mean([b.total_return_pct for b in benchmarks])
            with b1:
                st.metric("策略平均報酬", f"{total_return / len(res):+.1f}%")
            with b2:
                st.metric("買入持有報酬", f"{bh_return:+.1f}%")
            with b3:
                alpha = (total_return / len(res)) - bh_return
                color = "normal" if alpha >= 0 else "inverse"
                st.metric("超額報酬 (Alpha)", f"{alpha:+.1f}%", delta_color=color,
                           help="策略報酬 - 買入持有報酬。正數表示策略跑贏大盤")

        # Per-stock breakdown
        if show_detail:
            st.markdown("##### 📋 各股票表現")
            rows = []
            for code, r in res.items():
                pnl = r.final_value - r.initial_capital
                bh_return = r.benchmark.total_return_pct if r.benchmark else 0
                rows.append({
                    "股票代碼": code,
                    "初始資金": f"HKD {r.initial_capital:,.0f}",
                    "最終價值": f"HKD {r.final_value:,.0f}",
                    "盈虧": f"HKD {pnl:+,.0f}",
                    "報酬率": f"{r.total_return_pct:+.1f}%",
                    "買入持有": f"{bh_return:+.1f}%",
                    "勝率": f"{r.win_rate:.1f}%",
                    "Sharpe": f"{r.sharpe_ratio:.2f}",
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
                    "盈虧": f"HKD {t.pnl:+,.2f}" if t.action in ("Sell", "Hold") else "-",
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

    # --- Accuracy Dashboard ---
    st.markdown("---")
    st.subheader("🎯 預測準確度分析")
    st.caption("驗證歷史預測是否正確：Buy 信號後價格是否上漲？Sell 信號後價格是否下跌？")

    from src.model_monitoring import ModelDriftDetector
    from supabase import create_client

    try:
        client = create_client(SUPABASE_URL, SUPABASE_KEY)
        drift_detector = ModelDriftDetector(client)

        accuracy_data = []
        for code in selected_stocks:
            for tf, tf_label in [("1d", "1天"), ("5d", "5天"), ("20d", "20天")]:
                acc_result = drift_detector.calculate_accuracy(code, tf, days=60)
                if acc_result.get('accuracy') is not None and acc_result.get('total', 0) > 0:
                    accuracy_data.append({
                        "股票代碼": code,
                        "時間範圍": tf_label,
                        "準確度": f"{acc_result['accuracy']:.1f}%",
                        "正確/總數": f"{acc_result['correct']}/{acc_result['total']}",
                        "Buy 準確度": f"{acc_result['buy_accuracy']:.1f}%" if acc_result.get('buy_accuracy') is not None else "-",
                        "Sell 準確度": f"{acc_result['sell_accuracy']:.1f}%" if acc_result.get('sell_accuracy') is not None else "-",
                        "Buy 信號數": acc_result.get('buy_signals', 0),
                        "Sell 信號數": acc_result.get('sell_signals', 0),
                    })

        if accuracy_data:
            acc_df = pd.DataFrame(accuracy_data)
            st.dataframe(acc_df, use_container_width=True, hide_index=True)

            # Accuracy chart
            acc_chart_data = []
            for row in accuracy_data:
                acc_val = float(row["準確度"].replace("%", ""))
                acc_chart_data.append({
                    "股票": row["股票代碼"],
                    "時間範圍": row["時間範圍"],
                    "準確度": acc_val,
                })
            if acc_chart_data:
                chart_df = pd.DataFrame(acc_chart_data)
                fig_acc = px.bar(
                    chart_df,
                    x="股票",
                    y="準確度",
                    color="時間範圍",
                    barmode="group",
                    title="各股票各時間範圍預測準確度",
                    labels={"準確度": "準確度 (%)", "股票": "股票代碼"},
                )
                fig_acc.add_hline(y=50, line_dash="dash", line_color="gray", annotation_text="隨機基準 50%")
                fig_acc.update_layout(yaxis_range=[0, 100])
                st.plotly_chart(fig_acc, use_container_width=True, key="accuracy_chart")
        else:
            st.info("尚無足夠的歷史預測數據進行準確度分析。")

    except Exception as e:
        st.warning(f"無法載入準確度分析: {e}")

    # --- Portfolio Results ---
    if 'sim_portfolio' in st.session_state and st.session_state['sim_portfolio']:
        st.markdown("---")
        st.subheader("📊 組合模擬結果")
        portfolio = st.session_state['sim_portfolio']

        p1, p2, p3, p4 = st.columns(4)
        with p1:
            color = "normal" if portfolio.total_return_pct >= 0 else "inverse"
            st.metric("組合報酬", f"{portfolio.total_return_pct:+.1f}%", delta_color=color)
        with p2:
            st.metric("組合最終價值", f"HKD {portfolio.final_value:,.0f}")
        with p3:
            st.metric("組合 Sharpe", f"{portfolio.sharpe_ratio:.2f}")
        with p4:
            st.metric("組合最大回撤", f"{portfolio.max_drawdown_pct:.1f}%")

        if portfolio.portfolio_history:
            portfolio_df = pd.DataFrame([{
                "日期": s.date,
                "組合價值": s.portfolio_value,
            } for s in portfolio.portfolio_history])
            fig_port = px.line(
                portfolio_df, x="日期", y="組合價值",
                title="組合總價值走勢",
                labels={"組合價值": "價值 (HKD)", "日期": "日期"},
            )
            fig_port.add_hline(
                y=portfolio.initial_capital,
                line_dash="dash", line_color="gray",
                annotation_text=f"初始資金 HKD {portfolio.initial_capital:,.0f}",
            )
            fig_port.update_layout(yaxis_tickformat=",.0f")
            st.plotly_chart(fig_port, use_container_width=True, key="portfolio_chart")

    # --- Confidence Weighting Results ---
    if 'sim_confidence' in st.session_state and st.session_state['sim_confidence']:
        st.markdown("---")
        st.subheader("🎯 信心度加權策略結果")
        conf_res = st.session_state['sim_confidence']

        st.markdown("##### 信心度加權 vs 固定倉位")
        conf_rows = []
        for code, cr in conf_res.items():
            fixed = results.get(code)
            conf_rows.append({
                "股票代碼": code,
                "固定倉位報酬": f"{fixed.total_return_pct:+.1f}%" if fixed else "-",
                "信心度加權報酬": f"{cr.total_return_pct:+.1f}%",
                "固定倉位勝率": f"{fixed.win_rate:.1f}%" if fixed else "-",
                "信心度加權勝率": f"{cr.win_rate:.1f}%",
                "固定倉位 Sharpe": f"{fixed.sharpe_ratio:.2f}" if fixed else "-",
                "信心度加權 Sharpe": f"{cr.sharpe_ratio:.2f}",
            })
        st.dataframe(pd.DataFrame(conf_rows), use_container_width=True, hide_index=True)

    # --- Monte Carlo Results ---
    if 'sim_monte_carlo' in st.session_state and st.session_state['sim_monte_carlo']:
        st.markdown("---")
        st.subheader("🎲 蒙地卡羅測試結果")
        mc_res = st.session_state['sim_monte_carlo']

        for code, mc in mc_res.items():
            with st.expander(f"📊 {code} 蒙地卡羅分析"):
                st.markdown(f"**原始策略報酬:** {mc['original_return_pct']:+.1f}%")
                st.markdown(f"**翻轉概率:** {mc['flip_probability']:.0%} (每次信號有 {mc['flip_probability']:.0%} 機率被隨機翻轉)")

                mc1, mc2, mc3, mc4 = st.columns(4)
                with mc1:
                    st.metric("平均報酬", f"{mc['mean_return_pct']:+.1f}%")
                with mc2:
                    st.metric("中位數報酬", f"{mc['median_return_pct']:+.1f}%")
                with mc3:
                    st.metric("獲利機率", f"{mc['prob_profit']:.1f}%")
                with mc4:
                    st.metric("標準差", f"{mc['std_return_pct']:.1f}%")

                st.markdown("**報酬分佈:**")
                p1, p2, p3, p4, p5 = st.columns(5)
                with p1:
                    st.metric("5th percentile", f"{mc['percentile_5']:+.1f}%")
                with p2:
                    st.metric("25th percentile", f"{mc['percentile_25']:+.1f}%")
                with p3:
                    st.metric("中位數", f"{mc['median_return_pct']:+.1f}%")
                with p4:
                    st.metric("75th percentile", f"{mc['percentile_75']:+.1f}%")
                with p5:
                    st.metric("95th percentile", f"{mc['percentile_95']:+.1f}%")

                # Histogram of returns
                if mc.get('returns_distribution'):
                    hist_df = pd.DataFrame({"報酬率 (%)": mc['returns_distribution']})
                    fig_hist = px.histogram(
                        hist_df, x="報酬率 (%)", nbins=50,
                        title=f"{code} 蒙地卡羅報酬分佈 ({mc['n_simulations']} 次模擬)",
                        labels={"報酬率 (%)": "報酬率 (%)"},
                    )
                    fig_hist.add_vline(x=mc['original_return_pct'], line_dash="dash", line_color="red",
                                       annotation_text="原始策略")
                    fig_hist.add_vline(x=0, line_dash="dash", line_color="gray")
                    st.plotly_chart(fig_hist, use_container_width=True, key=f"mc_hist_{code}")

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
        - **勝率** = 盈利交易次數 ÷ 總交易次數 × 100% (包含賣出及持倉到期)
        - **Sharpe Ratio** = 年化風險調整報酬率 (>1 不錯, >2 很好, <0 不如無風險投資)
        - **Sortino Ratio** = 只考慮下跌風險的風險調整報酬率
        - **利潤因子** = 總盈利 / 總虧損 (>1 表示盈利大於虧損)
        - **平均持倉天數** = 每次買入到賣出的平均天數
        - **買入持有** = 基準策略：在開始時買入並持有到結束

        ### 進階功能

        | 功能 | 說明 |
        |------|------|
        | **組合模擬** | 模擬所有選中股票的組合表現，資金平均分配 |
        | **信心度加權** | 根據信號信心度調整倉位大小（高信心=大倉位，30%-100%） |
        | **蒙地卡羅測試** | 隨機翻轉信號1000次，測試策略穩健性，顯示報酬分佈 |
        | **預測準確度** | 驗證歷史預測是否正確：Buy後價格是否上漲？Sell後價格是否下跌？ |

        ### 注意事項

        - 本模擬僅供參考，不構成投資建議
        - 實際交易可能有滑點、流動性等影響
        - 不考慮做空（Sell 信號僅用於平倉）
        - 蒙地卡羅測試使用隨機模擬，結果可能每次不同
        """)

# --- Footer ---
st.markdown("---")
st.caption("⚠️ 本模擬僅供參考，不構成投資建議。投資有風險，入市需謹慎。")
