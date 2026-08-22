"""
Streamlit dashboard - reads predictions from Supabase and visualizes them.
Supports 3 timeframes: 1d (明日), 5d (下週), 20d (下月)
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

# Load .env from project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(PROJECT_ROOT, '.env'))

# Page config
st.set_page_config(
    page_title="港股 AI 預測儀表板",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Timeframe labels
TIMEFRAME_LABELS = {
    '1d': '📈 明日 (1天)',
    '5d': '📊 下週 (5天)',
    '20d': '📉 下月 (20天)'
}


# --- Supabase Connection ---
@st.cache_resource
def get_supabase_client():
    url = os.getenv('SUPABASE_URL')
    key = os.getenv('SUPABASE_KEY')
    if not url or not key:
        return None
    return create_client(url, key)


@st.cache_data(ttl=300)
def get_predictions(days=30):
    client = get_supabase_client()
    if client is None:
        return pd.DataFrame()
    start_date = (datetime.now() - timedelta(days=days)).date().isoformat()
    try:
        response = client.table('stock_predictions') \
            .select('*') \
            .gte('prediction_date', start_date) \
            .order('prediction_date', desc=True) \
            .execute()
        if response.data:
            return pd.DataFrame(response.data)
        return pd.DataFrame()
    except Exception:
        return pd.DataFrame()


# --- Main Page ---
st.title("📈 港股 AI 預測儀表板")

# Check connection
client = get_supabase_client()
if client is None:
    st.error("無法連線至資料庫。請檢查 .env 中的 SUPABASE_URL 和 SUPABASE_KEY。")
    st.stop()

# --- Sidebar Controls ---
days = st.slider("📅 選擇天數範圍", 7, 90, 30)
df = get_predictions(days)

if df.empty:
    st.info("暫無預測數據。請先執行:\n1. `python src/train_model.py` 訓練模型\n2. `python src/predict_upload.py` 生成預測")
    st.stop()

# Ensure proper types
df['prediction_date'] = pd.to_datetime(df['prediction_date']).dt.date
df['confidence'] = df['confidence'].astype(float)
df['stock_code'] = df['stock_code'].astype(str)
if 'timeframe' not in df.columns:
    df['timeframe'] = '1d'  # Legacy data

# --- Latest Signals for Each Timeframe ---
st.markdown("---")

for tf_label, tf_title in TIMEFRAME_LABELS.items():
    st.subheader(tf_title)
    st.caption("信心 = 模型預測上漲的機率。>55% → Buy，<45% → Sell，其餘 → Hold")

    tf_df = df[df['timeframe'] == tf_label]
    if tf_df.empty:
        st.info(f"暫無 {tf_label} 預測數據")
        continue

    # Get latest per stock
    latest = tf_df.sort_values('prediction_date').groupby('stock_code').last().reset_index()

    cols = st.columns(len(latest))
    for i, (_, row) in enumerate(latest.iterrows()):
        with cols[i]:
            signal = row['signal']
            if signal == 'Buy':
                emoji = "📈"
                delta_color = "normal"
            elif signal == 'Sell':
                emoji = "📉"
                delta_color = "inverse"
            else:
                emoji = "➡️"
                delta_color = "off"

            st.metric(
                label=f"{emoji} {row['stock_code']}",
                value=signal,
                delta=f"信心: {row['confidence']:.1%}",
                delta_color=delta_color
            )

    st.markdown("---")

# --- Confidence Trend (all timeframes) ---
st.subheader("📊 預測信心度趨勢")

fig_trend = px.line(
    df,
    x='prediction_date',
    y='confidence',
    color='stock_code',
    symbol='timeframe',
    markers=True,
    labels={
        'prediction_date': '日期',
        'confidence': '信心度',
        'stock_code': '股票代碼',
        'timeframe': '時間範圍'
    },
    title='各股票預測信心度變化'
)
fig_trend.add_hline(y=0.55, line_dash="dash", line_color="green", annotation_text="Buy 閾值")
fig_trend.add_hline(y=0.45, line_dash="dash", line_color="red", annotation_text="Sell 閾值")
fig_trend.update_layout(yaxis_tickformat='.0%')
st.plotly_chart(fig_trend, use_container_width=True)

# --- Signal Distribution per Timeframe ---
st.subheader("📋 信號分佈")
selected_tf = st.selectbox("選擇時間範圍", list(TIMEFRAME_LABELS.keys()),
                           format_func=lambda x: TIMEFRAME_LABELS[x])

tf_df = df[df['timeframe'] == selected_tf]
if not tf_df.empty:
    signal_counts = tf_df.groupby(['stock_code', 'signal']).size().reset_index(name='count')
    signal_counts['stock_code'] = signal_counts['stock_code'].astype(str)
    fig_pie = px.bar(
        signal_counts,
        x='stock_code',
        y='count',
        color='signal',
        color_discrete_map={'Buy': '#2ecc71', 'Sell': '#e74c3c', 'Hold': '#95a5a6'},
        labels={'stock_code': '股票代碼', 'count': '次數', 'signal': '信號'},
        title=f'{TIMEFRAME_LABELS[selected_tf]} 信號分佈'
    )
    fig_pie.update_xaxes(type='category')
    st.plotly_chart(fig_pie, use_container_width=True)

# --- Recent Predictions Table ---
st.markdown("---")
st.subheader("📝 近期預測記錄")

# Only select known columns, ignore extras (id, created_at, etc.)
base_cols = ['stock_code', 'prediction_date', 'timeframe', 'signal', 'confidence', 'model_version', 'created_at']
extra_cols = ['model_type', 'f1_score', 'auc_score', 'expected_return', 'risk_reward', 'stop_loss', 'take_profit', 'confidence_trend', 'win_rate']
available = [c for c in base_cols + extra_cols if c in df.columns]

display_df = df[available].copy()
display_df['timeframe'] = display_df['timeframe'].map(TIMEFRAME_LABELS)
display_df['信心度'] = display_df['confidence'].apply(lambda x: f"{x:.2%}")
display_df = display_df.drop(columns=['confidence'])

if 'f1_score' in display_df.columns:
    display_df['F1 分數'] = display_df['f1_score'].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "-")
if 'auc_score' in display_df.columns:
    display_df['AUC 分數'] = display_df['auc_score'].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "-")
if 'expected_return' in display_df.columns:
    display_df['預期報酬'] = display_df['expected_return'].apply(lambda x: f"{x:+.2f}%" if pd.notna(x) else "-")
if 'risk_reward' in display_df.columns:
    display_df['風險報酬比'] = display_df['risk_reward'].apply(lambda x: f"{x:.2f}" if pd.notna(x) and x > 0 else "-")
if 'stop_loss' in display_df.columns:
    display_df['止損'] = display_df['stop_loss'].apply(lambda x: f"{x:+.2f}%" if pd.notna(x) else "-")
if 'take_profit' in display_df.columns:
    display_df['止盈'] = display_df['take_profit'].apply(lambda x: f"{x:+.2f}%" if pd.notna(x) else "-")
if 'confidence_trend' in display_df.columns:
    display_df['趨勢'] = display_df['confidence_trend']
if 'win_rate' in display_df.columns:
    display_df['勝率'] = display_df['win_rate'].apply(lambda x: f"{x:.1f}%" if pd.notna(x) else "-")
if 'created_at' in display_df.columns:
    display_df['預測時間'] = pd.to_datetime(display_df['created_at']).dt.strftime('%Y-%m-%d %H:%M')

# Rename
rename_map = {
    'stock_code': '股票代碼',
    'prediction_date': '預測日期',
    'timeframe': '時間範圍',
    'signal': '信號',
    'model_version': '模型版本',
    'model_type': '冠軍模型',
}
display_df = display_df.rename(columns=rename_map)

# Final column order
final_cols = ['股票代碼', '預測日期', '時間範圍', '信號', '信心度', '趨勢', '預期報酬', '止損', '止盈', '風險報酬比', '勝率', '模型版本', '冠軍模型', 'F1 分數', 'AUC 分數', '預測時間']
final_cols = [c for c in final_cols if c in display_df.columns]
display_df = display_df[final_cols]
display_df = display_df.sort_values('預測時間', ascending=False)

st.dataframe(display_df, use_container_width=True, hide_index=True)

# Export functionality
st.markdown("---")
st.subheader("📥 匯出預測數據")

col1, col2 = st.columns(2)

with col1:
    # Export to CSV
    csv_data = display_df.to_csv(index=False, encoding='utf-8-sig')
    st.download_button(
        label="📄 匯出 CSV",
        data=csv_data,
        file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        help="下載 CSV 格式，可用 Excel 或 Google Sheets 開啟"
    )

with col2:
    # Export to Excel
    try:
        import openpyxl
        from io import BytesIO
        
        buffer = BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            display_df.to_excel(writer, index=False, sheet_name='預測記錄')
        
        st.download_button(
            label="📊 匯出 Excel",
            data=buffer.getvalue(),
            file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            help="下載 Excel 格式，包含格式化的工作表"
        )
    except ImportError:
        st.info("安裝 openpyxl 以啟用 Excel 匯出: `pip install openpyxl`")

# Model metrics explanation
if 'model_type' in df.columns:
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
    | **Stacking** | 同上三個基礎模型 + LogisticRegression 元模型 |
    | **SMOTE** | 訓練折上自動生成少數類合成樣本 |

    **模型表現 (F1 Score)：**
    | 時間範圍 | F1 Score | 說明 |
    |---|---|---|
    | 1天 | ~0.57 | 可用 — 短期趨勢 |
    | 5天 | ~0.69 | 良好 — 中期動量 |
    | 20天 | ~0.73 | 最佳 — 長期趨勢 |

    **信號判定：**
    | 信心度 | 信號 |
    |---|---|
    | > 55% | Buy (買入) |
    | < 45% | Sell (賣出) |
    | 45% ~ 55% | Hold (持有) |
    """)

# --- Model Monitoring Section ---
st.markdown("---")
st.subheader("🔍 模型監控")
st.caption("自動監控模型性能、數據品質和信號品質，確保預測可靠性")

# Import monitoring functions
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))
from model_monitoring import DataQualityChecker, ModelDriftDetector, AlertManager, ConfidenceCalibrator

# Initialize monitoring
checker = DataQualityChecker(client)
drift_detector = ModelDriftDetector(client)
alert_manager = AlertManager(client)
calibrator = ConfidenceCalibrator(client)

# Get stock codes from config
STOCK_LIST = os.getenv('STOCK_LIST', '0700,9988,0005,0939').split(',')

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
        missing_ok = result['missing_dates']['status'] == 'ok'
        dist_ok = result['confidence_dist']['status'] == 'ok'
        status = "✅" if missing_ok and dist_ok else "⚠️"
        
        st.write(f"{status} **{result['stock_code']}**")
        
        # Show missing dates status
        st.write(f"  - 數據完整性: {result['missing_dates']['message']}")
        if 'issues' in result['missing_dates']:
            for issue in result['missing_dates']['issues']:
                st.write(f"    - {issue}")
        
        # Show confidence distribution status
        st.write(f"  - 信心度分佈: {result['confidence_dist']['message']}")
        if 'issues' in result['confidence_dist']:
            for issue in result['confidence_dist']['issues']:
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
        if result.get('drift'):
            severity = "🔴" if result.get('severity') == 'high' else "🟡"
            st.write(f"{severity} **{result['stock_code']}** ({result['timeframe']}): {result['message']}")
        else:
            st.write(f"✅ **{result['stock_code']}** ({result['timeframe']}): {result['message']}")

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
        if calibration.get('avg_confidence') is not None:
            adjustment = calibrator.suggest_calibration_adjustment(calibration)
            calibration_data.append({
                'stock_code': code,
                'avg_confidence': calibration['avg_confidence'],
                'std_confidence': calibration.get('std_confidence', 0),
                'sample_size': calibration.get('sample_size', 0),
                'message': adjustment['message']
            })
    
    if calibration_data:
        for data in calibration_data:
            st.write(f"**{data['stock_code']}**: 平均信心度 {data['avg_confidence']:.1%} (標準差 {data['std_confidence']:.1%}) - {data['message']}")
            st.write(f"  - 數據量: {data['sample_size']} 筆預測")
    else:
        st.info("數據不足，無法進行信心度校準分析。")

# --- Footer ---
st.markdown("---")
st.caption("⚠️ 本系統僅供參考，不構成投資建議。投資有風險，入市需謹慎。")
