"""
Streamlit dashboard - reads predictions from Supabase and visualizes them.
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
    st.info("暫無預測數據。請先執行 `python src/train_model.py` 訓練模型，再執行 `python src/predict_upload.py` 生成預測。")
    st.stop()

# Ensure proper types
df['prediction_date'] = pd.to_datetime(df['prediction_date']).dt.date
df['confidence'] = df['confidence'].astype(float)

# --- Latest Signals (Top Section) ---
st.markdown("---")
st.subheader("🔮 當日最新信號")

# Get latest prediction per stock
latest = df.sort_values('prediction_date').groupby('stock_code').last().reset_index()

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

# --- Prediction Confidence Trend ---
st.subheader("📊 預測信心度趨勢")

fig_trend = px.line(
    df,
    x='prediction_date',
    y='confidence',
    color='stock_code',
    markers=True,
    labels={
        'prediction_date': '日期',
        'confidence': '信心度',
        'stock_code': '股票代碼'
    },
    title='各股票預測信心度變化'
)
fig_trend.add_hline(y=0.55, line_dash="dash", line_color="green", annotation_text="Buy 閾值")
fig_trend.add_hline(y=0.45, line_dash="dash", line_color="red", annotation_text="Sell 閾值")
fig_trend.update_layout(yaxis_tickformat='.0%')
st.plotly_chart(fig_trend, use_container_width=True)

# --- Signal Distribution ---
st.subheader("📋 信號分佈")
signal_counts = df.groupby(['stock_code', 'signal']).size().reset_index(name='count')
fig_pie = px.bar(
    signal_counts,
    x='stock_code',
    y='count',
    color='signal',
    color_discrete_map={'Buy': '#2ecc71', 'Sell': '#e74c3c', 'Hold': '#95a5a6'},
    labels={'stock_code': '股票代碼', 'count': '次數', 'signal': '信號'},
    title='各股票信號分佈'
)
st.plotly_chart(fig_pie, use_container_width=True)

# --- Recent Predictions Table ---
st.markdown("---")
st.subheader("📝 近期預測記錄")

display_df = df[['stock_code', 'prediction_date', 'signal', 'confidence', 'model_version']].copy()
display_df.columns = ['股票代碼', '預測日期', '信號', '信心度', '模型版本']
display_df['信心度'] = display_df['信心度'].apply(lambda x: f"{x:.2%}")
display_df = display_df.sort_values('預測日期', ascending=False)

st.dataframe(display_df, use_container_width=True, hide_index=True)

# --- Footer ---
st.markdown("---")
st.caption("⚠️ 本系統僅供參考，不構成投資建議。投資有風險，入市需謹慎。")
