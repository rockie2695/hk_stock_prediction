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
base_cols = ['stock_code', 'prediction_date', 'timeframe', 'signal', 'confidence', 'model_version']
extra_cols = ['model_type', 'f1_score', 'auc_score']
available = [c for c in base_cols + extra_cols if c in df.columns]

display_df = df[available].copy()
display_df['timeframe'] = display_df['timeframe'].map(TIMEFRAME_LABELS)
display_df['信心度'] = display_df['confidence'].apply(lambda x: f"{x:.2%}")
display_df = display_df.drop(columns=['confidence'])

if 'f1_score' in display_df.columns:
    display_df['F1 分數'] = display_df['f1_score'].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "-")
if 'auc_score' in display_df.columns:
    display_df['AUC 分數'] = display_df['auc_score'].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "-")

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
final_cols = ['股票代碼', '預測日期', '時間範圍', '信號', '信心度', '模型版本', '冠軍模型', 'F1 分數', 'AUC 分數']
final_cols = [c for c in final_cols if c in display_df.columns]
display_df = display_df[final_cols]
display_df = display_df.sort_values('預測日期', ascending=False)

st.dataframe(display_df, use_container_width=True, hide_index=True)

# Model metrics explanation
if 'model_type' in df.columns:
    with st.expander("📖 模型指標說明"):
        st.markdown("""
        | 指標 | 說明 | 範圍 |
        |---|---|---|
        | **F1 分數** | 精準率與召回率的調和平均數。越高表示模型預測越準確（兼顧「預測對的」和「不漏掉」）。0.5 為隨機水平，>0.6 為可用。 | 0 ~ 1 |
        | **AUC 分數** | 模型區分漲跌的能力。0.5 = 隨機猜測，1.0 = 完美區分。衡量模型對信心度排序的品質。 | 0.5 ~ 1.0 |
        | **冠軍模型** | Optuna 自動調參後，XGBoost 與 LightGBM 在驗證折上比較，選出 F1 較高者。 | xgboost / lightgbm |
        """)

# Features explanation
with st.expander("🔬 模型學習的技術指標 (Features)"):
    st.markdown("""
    模型使用 **3 年歷史數據** (約 750 交易日) 訓練，從 **OHLCV** + **市場指數** 計算以下 26 項特徵：

    | 類別 | 特徵名稱 | 說明 |
    |---|---|---|
    | **報酬率** | `ret_1d`, `ret_5d`, `ret_10d`, `ret_20d` | 1/5/10/20日漲跌幅 |
    | **價格形態** | `high_low_range`, `close_to_high`, `close_to_low` | 日內振幅、收盤位置 |
    | **成交量** | `vol_ratio_5d`, `vol_ratio_10d` | 量能相對強弱 |
    | **成交量** | `obv_change` | OBV (能量潮) 變化 |
    | **動量** | `rsi_14` | RSI 超買/超賣 |
    | **動量** | `stoch_k`, `stoch_d` | 隨機震盪指標 |
    | **動量** | `mfi` | 資金流量指標 |
    | **趨勢** | `macd_diff`, `macd_dea`, `macd_hist` | MACD 三元件 |
    | **趨勢** | `adx` | 趨勢強度 (不分方向) |
    | **波動** | `bb_width`, `atr_14` | 布林寬度、平均真實波幅 |
    | **統計** | `ret_5d_skew`, `ret_5d_kurt` | 報酬率偏度/峰度 |
    | **統計** | `volatility_10d` | 10日波動率 |
    | **市場** | `hsi_ret_5d`, `hsi_ret_20d` | 恒生指數漲跌幅 |
    | **匯率** | `usdhkd_change` | 美元/港幣匯率變化 |

    **目標變數 (Target)：**
    | 時間範圍 | 說明 |
    |---|---|
    | `1d` | 明日收盤 > 今日收盤 → 1 (上漲), 否則 → 0 |
    | `5d` | 5日後收盤 > 今日收盤 → 1, 否則 → 0 |
    | `20d` | 20日後收盤 > 今日收盤 → 1, 否則 → 0 |

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

# --- Footer ---
st.markdown("---")
st.caption("⚠️ 本系統僅供參考，不構成投資建議。投資有風險，入市需謹慎。")
