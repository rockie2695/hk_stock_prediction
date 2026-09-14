# 港股每日自動預測系統 / Hong Kong Stock Daily Auto-Prediction System

純機器學習的港股每日預測系統，使用 XGBoost / LightGBM / RandomForest / CatBoost 集成模型進行漲跌預測，結果自動上傳至 Supabase 雲端資料庫，並透過 Streamlit 網站展示。
Pure ML-based daily HK stock prediction system using XGBoost / LightGBM / RandomForest / CatBoost ensemble models for up/down prediction. Results are auto-uploaded to Supabase cloud database and displayed via Streamlit dashboard.

> **Warning:** 本系統僅供學習和研究使用，不構成任何投資建議。投資有風險，入市需謹慎。 / This system is for educational and research purposes only. It does not constitute investment advice. Investing involves risk — invest with caution.

## 系統架構 / System Architecture

```
Windows 本地定時訓練 (平行) → 預測結果上傳至 Supabase (PostgreSQL) → Streamlit 網站顯示
Local Windows scheduled training (parallel) → Upload predictions to Supabase (PostgreSQL) → Streamlit website display
```

## 功能特色 / Features

### 核心功能 / Core Features
- **多時間範圍預測**: 同時預測明日(1天)、下週(5天)、下月(20天) / **Multi-timeframe Prediction**: Predicts tomorrow (1-day), next week (5-day), next month (20-day) simultaneously
- **四模型集成**: XGBoost + LightGBM + RandomForest + CatBoost / **Four-model Ensemble**: XGBoost + LightGBM + RandomForest + CatBoost
- **三種集成模式**: Voting (加權平均) / Stacking (元模型) / Blending (out-of-fold) / **Three Ensemble Modes**: Voting (weighted average) / Stacking (meta-model) / Blending (out-of-fold)
- **平行訓練**: 時間範圍同時訓練，速度提升 ~3x / **Parallel Training**: Timeframes trained simultaneously, ~3x speedup
- **平行預測**: 多支股票同時預測 / **Parallel Prediction**: Multiple stocks predicted concurrently
- **SMOTE 類別平衡**: 自動處理正負樣本不平衡問題 / **SMOTE Class Balancing**: Automatically handles positive/negative class imbalance
- **33項技術指標**: 新增動量、波動率、威廉指標、MFI 等 / **33 Technical Indicators**: Includes momentum, volatility, Williams %R, MFI, etc.
- **15項擴展特徵**: 情緒分析、板塊輪動、沽空比率、互聯互通資金流、市場狀態偵測 / **15 Extended Features**: Sentiment analysis, sector rotation, short selling, connect flow, market regime detection
- **特徵相關性過濾**: 自動移除 |corr| > 0.9 的冗餘特徵 / **Feature Correlation Filter**: Auto-removes redundant features with |corr| > 0.9
- **閾值優化**: 自動搜尋最佳 Buy/Sell 信心度閾值 (取代固定 0.55/0.45) / **Threshold Optimization**: Auto-searches optimal Buy/Sell confidence thresholds (replaces fixed 0.55/0.45)
- **模型版本化**: 帶時間戳的模型備份，自動保留最近 5 版，支持回滾 / **Model Versioning**: Timestamped model backups, keeps latest 5 versions, supports rollback
- **模型指標追蹤**: 記錄 F1 Score、AUC Score、冠軍模型類型 / **Model Metrics Tracking**: Records F1 Score, AUC Score, champion model type
- **模型分歧檢測**: 當模型意見分歧 >= 50% 時強制 Hold，顯示分歧程度 / **Model Disagreement Detection**: Forces Hold when model disagreement >= 50%, displays disagreement level
- **GPU 支援**: CatBoost 可選擇使用 GPU 加速 (透過 `USE_GPU=True` 啟用) / **GPU Support**: CatBoost can optionally use GPU acceleration (enable via `USE_GPU=True`)
- **互動式儀表板**: Streamlit 顯示預測結果、信心度趨勢、信號分佈 / **Interactive Dashboard**: Streamlit displays prediction results, confidence trends, signal distribution
- **K線圖 (含買賣信號)**: 互動式K線圖，標示模型預測的買入/賣出信號位置，支援自選股票和時間範圍 / **K-line Chart (with Buy/Sell Signals)**: Interactive K-line chart marking model-predicted buy/sell signal positions, supports custom stock and timeframe selection
- **MA均線疊加**: K線圖可疊加顯示 MA5 (短期)、MA10 (中期)、MA20 (長期) 移動平均線 / **MA Overlay**: K-line chart can overlay MA5 (short-term), MA10 (mid-term), MA20 (long-term) moving averages
- **股票對比圖**: 選擇兩支股票並排比較累計報酬率走勢和表現指標 / **Stock Comparison Chart**: Select two stocks to compare cumulative return trends and performance metrics side-by-side
- **技術指標展示**: 顯示 RSI、MACD、Stochastic、ADX、MFI、布林帶寬、ATR、量比等最新數值，作為信號依據 / **Technical Indicator Display**: Shows latest RSI, MACD, Stochastic, ADX, MFI, Bollinger Band Width, ATR, Volume Ratio values as signal basis
- **信號確認分析**: 每個 Buy/Sell 信號自動分析指標 alignment (支持/矛盾/中性)，幫助判斷信號可靠性 / **Signal Confirmation Analysis**: Each Buy/Sell signal auto-analyzes indicator alignment (support/contradict/neutral) to help assess signal reliability
- **閾值互動控制**: 圖表可選擇時間範圍顯示 Buy/Sell 閾值線，避免多線重疊 / **Threshold Interactive Control**: Charts can show Buy/Sell threshold lines for selected timeframes, avoiding line overlap
- **數據匯出**: 支援 CSV 和 Excel 格式匯出預測記錄 / **Data Export**: Supports CSV and Excel format export of prediction records
- **投資模擬器**: 根據 Buy/Sell 信號模擬投資，計算實際收益、勝率、最大回撤，支援自訂資金與日期範圍 / **Investment Simulator**: Simulates investment based on Buy/Sell signals, calculates actual returns, win rate, max drawdown, supports custom capital and date range
- **風險指標**: Sharpe Ratio、Sortino Ratio、利潤因子、平均持倉天數 / **Risk Metrics**: Sharpe Ratio, Sortino Ratio, Profit Factor, Average Holding Days
- **基準對比**: 策略報酬 vs 買入持有 (Buy & Hold) 對比，顯示超額報酬 (Alpha) / **Benchmark Comparison**: Strategy return vs Buy & Hold, displays excess return (Alpha)
- **預測準確度分析**: 驗證歷史預測是否正確，顯示各股票各時間範圍的真實準確度 / **Prediction Accuracy Analysis**: Validates whether historical predictions were correct, shows true accuracy per stock and timeframe
- **組合模擬**: 多股票組合模擬，顯示組合表現和 diversification 效果 / **Portfolio Simulation**: Multi-stock portfolio simulation, shows portfolio performance and diversification effects
- **信心度加權策略**: 根據信號信心度調整倉位大小 (高信心=大倉位) / **Confidence-Weighted Strategy**: Adjusts position size based on signal confidence (high confidence = larger position)
- **蒙地卡羅測試**: 隨機翻轉信號，測試策略穩健性，顯示報酬分佈 (可自訂模擬次數 100-5000) / **Monte Carlo Test**: Randomly flips signals to test strategy robustness, shows return distribution (customizable 100-5000 simulations)
- **HK 公曆日曆**: 預測日期自動跳過港股休市日，從官方 1823.gov.hk API 即時獲取假期數據 (含農曆節日) / **HK Public Holiday Calendar**: Prediction dates auto-skip HK market holidays, fetches holiday data from official 1823.gov.hk API (including Lunar New Year)
- **滑點模擬**: 模擬實際交易滑點，買入價格略高、賣出價格略低 (可調 0-1%) / **Slippage Simulation**: Simulates real trading slippage — buy price slightly higher, sell price slightly lower (adjustable 0-1%)
- **板手交易**: 按港股最低交易單位 (board lot) 計算買入股數，更貼近實際交易 / **Board Lot Trading**: Calculates buy quantity per HK minimum trading unit (board lot), closer to real trading
- **止損/止盈執行**: 根據預測的止損/止盈水平自動平倉 (在信號日之間檢查每日價格) / **Stop Loss/Take Profit Execution**: Auto-closes positions based on predicted stop loss/take profit levels (checks daily prices between signal dates)
- **本地數據快取**: 使用 Parquet 格式快取歷史數據，交易日內 4 小時快取、非交易日 24 小時快取 / **Local Data Cache**: Uses Parquet format for caching historical data — 4-hour cache on trading days, 24-hour cache on non-trading days

### 風險管理 / Risk Management
- **止損/止盈建議**: 基於波動率自動計算建議止損止盈點 / **Stop Loss/Take Profit Recommendations**: Auto-calculates recommended stop loss and take profit levels based on volatility
- **風險報酬比**: 評估潛在收益與風險的比例 / **Risk-Reward Ratio**: Evaluates the ratio of potential gain to risk
- **預期報酬**: 基於信心度和波動率估算預期報酬率 / **Expected Return**: Estimates expected return rate based on confidence and volatility
- **信心度追蹤**: 顯示信心度變化趨勢 (↑↓→) / **Confidence Tracking**: Shows confidence change trend (↑↓→)
- **勝率統計**: 歷史預測準確率追蹤 / **Win Rate Statistics**: Historical prediction accuracy tracking
- **模型分歧檢測**: 當模型意見分歧時自動 Hold，避免弱勢決策 / **Model Disagreement Detection**: Auto-Hold when models disagree, avoiding weak decisions

### 模型監控 / Model Monitoring
- **數據品質檢查**: 自動檢測缺失日期、信心度分佈異常 / **Data Quality Checks**: Auto-detects missing dates, abnormal confidence distribution
- **真實準確度驗證**: 透過實際價格數據驗證預測是否正確 (非僅信心度代理) / **True Accuracy Verification**: Validates predictions using actual price data (not just confidence as proxy)
- **滾動準確度追蹤**: 30天滾動窗口計算真實預測準確度 (Buy後價格是否上漲？Sell後價格是否下跌？) / **Rolling Accuracy Tracking**: 30-day rolling window for true prediction accuracy (Did price rise after Buy? Fall after Sell?)
- **訓練指標展示**: 顯示各股票各時間範圍的 F1 Score、AUC Score、冠軍模型類型 / **Training Metrics Display**: Shows F1 Score, AUC Score, champion model type for each stock and timeframe
- **模型漂移檢測**: 監控模型性能是否下降 (>10% = 高度, >5% = 中度) / **Model Drift Detection**: Monitors model performance degradation (>10% = high, >5% = moderate)
- **回測引擎**: 實際模擬策略表現，計算真實回報、勝率、Sharpe Ratio / **Backtest Engine**: Actually simulates strategy performance, calculates real returns, win rate, Sharpe Ratio
- **信號警報**: 強勢信號自動提醒 (信心度>70% 或 預期報酬>5%) / **Signal Alerts**: Auto-alerts for strong signals (confidence >70% or expected return >5%)
- **信心度校準**: 確保信心度分數可靠 / **Confidence Calibration**: Ensures confidence scores are reliable

## 快速開始 / Quick Start

### 1. 安裝依賴 / Install Dependencies

```bash
# 方法一：使用批次檔一鍵安裝 / Method One: One-click install via batch file
setup.bat

# 方法二：手動安裝 / Method Two: Manual installation
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 2. 配置環境變數 / Configure Environment Variables

1. 複製 `.env.example` 為 `.env` / Copy `.env.example` to `.env`
2. 填入你的 Supabase 專案資訊 / Fill in your Supabase project info:

```env
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-anon-key
STOCK_LIST=0700,9988,0005,0939

# 模型訓練開關 / Model training toggles
USE_ENSEMBLE=True
USE_STACKING=False
USE_BLENDING=False
USE_CATBOOST=True
USE_SMOTE=True
USE_GPU=False

# 擴展特徵開關 / Extended feature toggles
USE_SENTIMENT=True
USE_SECTOR=True
USE_SHORT_SELL=True
USE_CONNECT=True
USE_REGIME=True
USE_ONLINE_LEARNING=False
USE_DYNAMIC_WEIGHTING=False
```

**⚠️ 重要 / IMPORTANT:** 先在 [Supabase 官網](https://supabase.com) 取得專案 URL 與金鑰，填入 `.env` 後再執行。 / Obtain your project URL and key from the [Supabase website](https://supabase.com), fill in `.env`, then proceed.

> **Warning:** 絕對不要將 `.env` 檔案提交至版本控制 (Git)。該檔案包含 Supabase 金鑰等敏感資訊。請確認 `.gitignore` 中已包含 `.env`。 / NEVER commit `.env` files to version control (Git). This file contains sensitive info like Supabase keys. Verify `.env` is in `.gitignore`.

### 3. 初始化資料庫 / Initialize Database

```bash
python src/init_database.py
```

### 4. 訓練模型 / Train Model

```bash
python src/train_model.py
```

訓練完成後會顯示： / After training completes, you will see:
- 各時間範圍的冠軍模型 (xgboost/lightgbm/catboost/voting/stacking/blending) / Champion model per timeframe (xgboost/lightgbm/catboost/voting/stacking/blending)
- 模型比較表 (各模型 F1 分數) / Model comparison table (F1 scores)
- F1 Score 和 AUC Score / F1 Score and AUC Score
- Top 10 特徵重要性 / Top 10 feature importance

### 5. 每日預測與上傳 / Daily Prediction & Upload

```bash
python src/predict_upload.py
```

### 6. 啟動預測儀表板 / Launch Prediction Dashboard

```bash
streamlit run app/streamlit_app.py
```

儀表板包含兩個頁面 (側邊欄切換)： / Dashboard contains two pages (sidebar toggle):
- **📈 預測儀表板 / Prediction Dashboard**: 分頁式設計，包含： / Tabbed design, includes:
  - **📊 信號總覽 / Signal Overview**: 信號卡片 (1d/5d/20d) + 指標 alignment 分析 + 信號分佈 / Signal cards (1d/5d/20d) + indicator alignment analysis + signal distribution
  - **📈 K線與指標 / K-line & Indicators**: K線圖 (含買賣信號) + 技術指標 + 股票對比 / K-line chart (with buy/sell signals) + technical indicators + stock comparison
  - **🎯 信心度趨勢 / Confidence Trend**: 信心度變化圖表 + Buy/Sell 閾值線 / Confidence change chart + Buy/Sell threshold lines
  - **📋 預測記錄 / Prediction Records**: 近期預測表格 + 匯出功能 / Recent prediction table + export functionality
  - **🔍 模型表現 / Model Performance**: 滾動準確度 + 訓練指標 + 模型監控 + 市場狀態 / Rolling accuracy + training metrics + model monitoring + market regime
- **💰 投資模擬器 / Investment Simulator**: 自訂日期範圍、資金、時間範圍，模擬跟單收益 / Custom date range, capital, timeframe, simulates copy-trading returns
- **📊 策略回測 / Strategy Backtest**: 資金曲線、回撤、交易記錄、對比買入持有 / Equity curve, drawdown, trade log, vs buy-and-hold
- **💼 投資組合 / Portfolio**: 持倉總覽、信號/信心度分佈、風險暴露 / Holdings, signal/confidence distribution, risk exposure

### 7. 執行測試 / Run Tests

```bash
# 執行所有測試 / Run all tests
python -m pytest tests/ -v

# 執行特定測試檔案 / Run specific test file
python -m pytest tests/test_config.py -v

# 執行特定測試類別 / Run specific test class
python -m pytest tests/test_feature_engineering.py::TestFeatureEngineering -v

# 執行特定測試函數 / Run specific test function
python -m pytest tests/test_train_model.py::TestModelTraining::test_train_xgboost -v

# 顯示詳細資訊 / Show verbose output
python -m pytest tests/ -v --tb=long

# 只顯示失敗的測試 / Show only failed tests
python -m pytest tests/ -v --tb=short
```

**測試覆蓋範圍 / Test Coverage：**
| 測試檔案 / Test File | 測試數量 / Tests | 覆蓋範圍 / Coverage |
|---|---|---|
| `test_config.py` | 10 | 環境變數、股票列表解析、Supabase 設定 / Env vars, stock list parsing, Supabase config |
| `test_feature_engineering.py` | 17 | RSI、MACD、Bollinger、ATR、ADX、Stochastic、MFI、Williams %R |
| `test_train_model.py` | 14 | XGBoost、LightGBM、RandomForest、CatBoost (含 early stopping)、SMOTE、Blending |
| `test_predict.py` | 10 | 預測日期、模型載入、信號判定、上傳功能 / Prediction dates, model loading, signal determination, upload |
| `test_sentiment.py` | 4 | 新聞情緒特徵計算 / News sentiment feature computation |
| `test_sector.py` | 4 | 板塊輪動特徵計算 / Sector rotation feature computation |
| `test_short_selling.py` | 4 | 沽空比率特徵計算 / Short selling feature computation |
| `test_connect_flow.py` | 4 | 互聯互通資金流特徵計算 / Connect flow feature computation |
| `test_regime.py` | 4 | 市場狀態偵測 / Market regime detection |
| `test_online_learner.py` | 3 | 增量學習 / Online learning |
| `test_dynamic_weighting.py` | 5 | 動態集成權重 / Dynamic ensemble weighting |
| `test_backtest.py` | 2 | 回測頁面 / Backtest page |
| `test_portfolio.py` | 2 | 投資組合頁面 / Portfolio page |
| **總計 / Total** | **85** | |

## 設定 Windows 自動排程 / Windows Task Scheduler Setup

使用 Windows 工作排程器，設定每日 16:30（港股收盤後）自動執行：
Set up Windows Task Scheduler to auto-execute daily at 16:30 (after HK market close):

1. 按 `Win + R`，輸入 `taskschd.msc` 開啟工作排程器 / Press `Win + R`, enter `taskschd.msc` to open Task Scheduler
2. 點擊右側「建立基本工作...」 / Click "Create Basic Task..." on the right
3. 名稱：`港股每日預測` / Name: `HK Stock Daily Prediction`
4. 觸發器：選擇「每日」，開始時間設為 `16:30` / Trigger: Select "Daily", set start time to `16:30`
5. 動作：選擇「啟動程式」 / Action: Select "Start a program"
6. 程式或指令：瀏覽選擇 `run_daily.bat` / Program or script: Browse and select `run_daily.bat`
7. 完成後，右鍵該工作 → 內容 → 設定： / After completion, right-click the task → Properties → Settings:
   - ✅ 喚醒電腦執行此工作 / Wake computer to run this task
   - ✅ 不論使用者是否登入都要執行 / Run whether user is logged on or not

> **Note:** 港股市場交易時間為 9:30-16:00 HKT。建議排程設定在 16:30，確保收盤數據已完全載入。週六、日及公眾假期為休市日，系統會自動跳過。 / HK market trading hours are 9:30-16:00 HKT. Schedule at 16:30 to ensure closing data is fully loaded. Weekends and public holidays are non-trading days — the system auto-skips them.

## Docker 部署 / Docker Deployment

### 建立鏡像 / Build Image

```bash
docker build -t hk-stock-prediction .
```

### 執行儀表板 / Run Dashboard

```bash
docker run -d -p 8501:8501 --env-file .env --name stock-dashboard hk-stock-prediction
```

開啟瀏覽器訪問 `http://localhost:8501` / Open browser and visit `http://localhost:8501`

### 執行訓練 (一次性) / Run Training (one-time)

```bash
docker run --env-file .env hk-stock-prediction python src/train_model.py
```

### 執行每日預測 / Run Daily Prediction

```bash
docker run --env-file .env hk-stock-prediction python src/predict_upload.py
```

### Windows 自動排程 (Docker) / Windows Task Scheduler (Docker)

使用 Windows 工作排程器，每日 16:30 自動執行 Docker 容器：
Use Windows Task Scheduler to auto-run Docker container daily at 16:30:

```bash
docker run --env-file C:\hk_stock_prediction\.env hk-stock-prediction python src/predict_upload.py
```

### 參數說明 / Parameter Description

| 參數 / Parameter | 說明 / Description |
|---|---|
| `-d` | 背景運行 / Run in background |
| `-p 8501:8501` | 映射 Streamlit 預設埠 / Maps Streamlit default port |
| `--env-file .env` | 載入環境變數 / Load environment variables |
| `--name stock-dashboard` | 容器名稱 / Container name |

> **Warning:** 絕對不要在 Docker 鏡像中硬編碼 Supabase 金鑰或任何 `.env` 內容。使用 `--env-file` 在運行時注入環境變數。切勿將 `.env` 檔案 COPY 到鏡像內部。 / NEVER hardcode Supabase keys or any `.env` content inside Docker images. Use `--env-file` to inject environment variables at runtime. Never COPY `.env` files into the image.

> **Note:** 使用 GPU 加速 (CatBoost) 時，Docker 需要額外安裝 NVIDIA Container Toolkit。確保宿主機已安裝 NVIDIA 驅動程式，並使用 `--gpus all` 參數運行容器。 / When using GPU acceleration (CatBoost), Docker requires NVIDIA Container Toolkit. Ensure the host machine has NVIDIA drivers installed and run the container with `--gpus all`.

## 專案結構 / Project Structure

```
project_root/
├── .env.example          # 環境變數範例 / Environment variable template
├── .gitignore            # Git 忽略清單 / Git ignore list
├── .dockerignore         # Docker 忽略清單 / Docker ignore list
├── Dockerfile            # Docker 鏡像定義 / Docker image definition
├── requirements.txt      # Python 依賴 / Python dependencies
├── config.py             # 讀取 .env，提供全域設定 / Reads .env, provides global config
├── run_daily.bat         # Windows 批次檔 (排程器用) / Windows batch file (for scheduler)
├── setup.bat             # 一鍵安裝依賴 / One-click dependency install
├── logs/                 # 日誌資料夾 / Log folder
├── models/               # 訓練好的模型 (.pkl) / Trained models (.pkl)
│   ├── best_model_{tf}.pkl           # 當前模型 / Current model
│   ├── best_model_{tf}_{ts}.pkl      # 版本化模型 (保留最近5版) / Versioned models (keeps latest 5)
│   ├── feature_importance_{tf}.csv   # 特徵重要性 / Feature importance
│   └── roc_curve_{tf}.png            # ROC 曲線 / ROC curve
├── tests/                # 單元測試 / Unit tests
│   ├── __init__.py
│   ├── test_config.py           # 設定模組測試 / Config module tests
│   ├── test_feature_engineering.py  # 特徵工程測試 / Feature engineering tests
│   ├── test_train_model.py      # 模型訓練測試 / Model training tests
│   └── test_predict.py          # 預測上傳測試 / Prediction upload tests
├── src/
│   ├── __init__.py
│   ├── logger.py         # 日誌設定 / Logger configuration
│   ├── init_database.py  # 自動建表 (冪等) / Auto-create tables (idempotent)
│   ├── data_fetcher.py   # 下載港股歷史數據 (akshare/yfinance) / Download HK stock data (akshare/yfinance)
│   ├── feature_engineering.py  # 48項技術指標計算 (33基礎+15擴展) / 48 technical indicators (33 base + 15 extended)
│   ├── train_model.py    # Optuna 自動調參 + Voting/Stacking 集成 + SMOTE / Optuna tuning + Voting/Stacking ensemble + SMOTE
│   ├── predict_upload.py # 每日預測並上傳 Supabase / Daily prediction & upload to Supabase
│   ├── simulator.py      # 投資模擬引擎 (信號模擬、交易成本、績效追蹤) / Investment simulator (signal simulation, costs, performance tracking)
│   ├── cleanup_old.py    # 清理舊數據 (保留60天) / Cleanup old data (keeps 60 days)
│   ├── model_monitoring.py  # 數據品質、模型漂移、警報、校準 / Data quality, model drift, alerts, calibration
│   ├── sentiment.py      # 新聞情緒特徵 / News sentiment features
│   ├── sector.py         # 板塊輪動特徵 / Sector rotation features
│   ├── short_selling.py  # 沽空比率特徵 / Short selling features
│   ├── connect_flow.py   # 互聯互通資金流特徵 / Connect flow features
│   ├── regime.py         # 市場狀態偵測 (牛/熊/震盪) / Market regime detection (bull/bear/sideways)
│   ├── online_learner.py # 增量學習 (warm-start) / Online learning (warm-start)
│   └── dynamic_weighting.py  # 動態集成權重 / Dynamic ensemble weighting
├── app/
│   ├── __init__.py
│   ├── streamlit_app.py  # Streamlit 預測儀表板 / Streamlit prediction dashboard
│   └── pages/
│       ├── __init__.py
│       ├── 1_💰_投資模擬器.py  # 投資模擬互動頁面 / Investment simulator page
│       ├── 2_backtest.py      # 策略回測頁面 / Strategy backtest page
│       └── 3_portfolio.py     # 投資組合頁面 / Portfolio page
├── migrate_metrics.sql   # 資料庫遷移: 模型指標欄位 / DB migration: model metrics fields
├── migrate_quick_wins.sql # 資料庫遷移: 風險管理欄位 / DB migration: risk management fields
├── migrate_thresholds.sql # 資料庫遷移: Buy/Sell 閾值欄位 / DB migration: Buy/Sell threshold fields
├── migrate_disagreement.sql # 資料庫遷移: 模型分歧指標 / DB migration: model disagreement metrics
└── migrations/
    └── 003_extended_features.sql # 擴展特徵說明 (無需資料庫變更) / Extended features docs (no DB changes)
```

> **Warning:** `.env` 和 `.pkl` 檔案不應提交至版本控制。`.gitignore` 已配置忽略這些檔案。如果意外提交，請立即從 Git 歷史中清除。 / `.env` and `.pkl` files must NOT be committed to version control. `.gitignore` is configured to ignore them. If accidentally committed, remove them from Git history immediately.

## 技術細節 / Technical Details

### 機器學習模型 / Machine Learning Models
- **演算法**: XGBoost + LightGBM + RandomForest + CatBoost 集成 / **Algorithms**: XGBoost + LightGBM + RandomForest + CatBoost ensemble
- **集成方式**: VotingClassifier (soft voting) 或 StackingClassifier (元模型 = LogisticRegression) 或 Blending (out-of-fold) / **Ensemble Methods**: VotingClassifier (soft voting) or StackingClassifier (meta-model = LogisticRegression) or Blending (out-of-fold)
- **超參數優化**: Optuna (50 trials，同時搜尋四個模型 + voting 權重) / **Hyperparameter Optimization**: Optuna (50 trials, simultaneously searching 4 models + voting weights)
- **權重優化**: Optuna 自動搜尋最佳權重組合 (如 [0.3, 0.3, 0.2, 0.2])，非固定 1:1:1:1 / **Weight Optimization**: Optuna auto-searches optimal weight combination (e.g. [0.3, 0.3, 0.2, 0.2]), not fixed 1:1:1:1
- **交叉驗證**: TimeSeriesSplit (n_splits=5)，嚴格遵守時序，不洩漏未來資訊 / **Cross-Validation**: TimeSeriesSplit (n_splits=5), strictly follows time order, no future data leakage
- **類別不平衡處理**: SMOTE (僅在訓練折上套用，不跨越驗證折) / **Class Imbalance Handling**: SMOTE (applied only on training folds, never across validation folds)
- **訓練數據**: 3 年歷史數據 (約 750 交易日) / **Training Data**: 3 years of historical data (~750 trading days)
- **評估指標**: F1 Score, AUC, Precision, Recall / **Evaluation Metrics**: F1 Score, AUC, Precision, Recall
- **ROC 曲線**: 自動儲存至 `models/roc_curve_{timeframe}.png` / **ROC Curve**: Auto-saved to `models/roc_curve_{timeframe}.png`
- **特徵相關性過濾**: 自動移除 |corr| > 0.9 的冗餘特徵 / **Feature Correlation Filter**: Auto-removes redundant features with |corr| > 0.9
- **閾值優化**: 自動搜尋最佳 Buy/Sell 信心度閾值 (取代固定 0.55/0.45) / **Threshold Optimization**: Auto-searches optimal Buy/Sell confidence thresholds (replaces fixed 0.55/0.45)
- **模型版本化**: 帶時間戳備份，自動保留最近 5 版 / **Model Versioning**: Timestamped backups, auto-keeps latest 5 versions
- **特徵重要性**: 輸出至 `models/feature_importance_{timeframe}.csv` / **Feature Importance**: Output to `models/feature_importance_{timeframe}.csv`
- **模型分歧檢測**: 當四個模型意見分歧 >= 50% 時強制 Hold / **Model Disagreement**: Forces Hold when 4 models disagree >= 50%
- **特徵對齊**: 訓練時保存 `feature_columns`，預測時嚴格使用相同順序 / **Feature Alignment**: Saves `feature_columns` during training, strictly uses same order during prediction
- **CatBoost 早停**: CatBoost 使用 early_stopping_rounds=30，搭配 eval_set 驗證集，自動停止訓練避免過擬合 (iterations 200-300) / **CatBoost Early Stopping**: Uses early_stopping_rounds=30 with eval_set validation, auto-stops training to prevent overfitting (iterations 200-300)
- **GPU 支援**: CatBoost 可選擇使用 GPU 加速 (透過 `USE_GPU=True` 啟用) / **GPU Support**: CatBoost can optionally use GPU acceleration (enable via `USE_GPU=True`)

> **Note:** 特徵工程嚴禁 Look-ahead Bias。所有技術指標僅使用當天及之前數據計算，絕不使用未來資訊。任何特徵計算違規都將導致模型虛高表現，在實際預測中失效。 / Feature engineering MUST NOT contain Look-ahead Bias. All technical indicators are computed using only data from the current day and earlier — never future data. Any feature calculation violation will cause inflated model performance that fails in real prediction.

> **Warning:** SMOTE 僅在 TimeSeriesSplit 的訓練折 (training fold) 上套用，絕對不能跨越驗證折。這是防止數據洩漏的關鍵措施。如果 SMOTE 在整個訓練集上套用，模型會在驗證集上獲得虛高的 AUC。 / SMOTE is applied ONLY on training folds of TimeSeriesSplit, never across validation folds. This is critical to prevent data leakage. If SMOTE is applied on the entire training set, the model will achieve inflated AUC on validation.

> **Note:** GPU 加速需要 NVIDIA GPU 及已安裝的 NVIDIA 驅動程式。CatBoost GPU 訓練會使用更多記憶體，建議至少 4GB VRAM。非 NVIDIA GPU (如 AMD) 不支援。 / GPU acceleration requires an NVIDIA GPU and installed NVIDIA drivers. CatBoost GPU training uses more memory — at least 4GB VRAM recommended. Non-NVIDIA GPUs (e.g., AMD) are not supported.

### 技術指標 (48 Features) / Technical Indicators (48 Features)

**基礎技術指標 (33 Features) / Base Technical Indicators (33 Features)：**

| 類別 / Category | 特徵 / Feature | 說明 / Description |
|---|---|---|
| **報酬率 / Returns** | `ret_1d`, `ret_3d`, `ret_5d`, `ret_10d`, `ret_20d`, `ret_30d` | 1/3/5/10/20/30日漲跌幅 / 1/3/5/10/20/30-day price change |
| **價格形態 / Price Pattern** | `high_low_range`, `close_to_high`, `close_to_low` | 日內振幅、收盤位置 / Intraday range, closing position |
| **價格位置 / Price Position** | `ma50_deviation` | 當前價格與 50 日均線乖離率 / Current price deviation from 50-day MA |
| **成交量 / Volume** | `vol_ratio_5d`, `vol_ratio_10d` | 量能相對強弱 / Relative volume strength |
| **成交量 / Volume** | `obv_change` | OBV (能量潮) 變化 / OBV (On-Balance Volume) change |
| **成交量 / Volume** | `volume_cv` | 成交量變異係數 (20日) / Volume coefficient of variation (20-day) |
| **動量 / Momentum** | `rsi_14` | RSI 超買/超賣 / RSI overbought/oversold |
| **動量 / Momentum** | `stoch_k`, `stoch_d` | 隨機震盪指標 / Stochastic oscillator |
| **動量 / Momentum** | `mfi` | 資金流量指標 / Money Flow Index |
| **動量 / Momentum** | `williams_r` | 威廉指標 (%R) / Williams %R |
| **趨勢 / Trend** | `macd_diff`, `macd_dea`, `macd_hist` | MACD 三元件 / MACD three components |
| **趨勢 / Trend** | `adx` | 趨勢強度 (不分方向) / Trend strength (direction-agnostic) |
| **波動 / Volatility** | `bb_width` | 布林通道寬度 / Bollinger Band width |
| **波動 / Volatility** | `atr_14`, `atr_ratio` | 平均真實波幅、ATR/收盤價比值 / Average True Range, ATR/close ratio |
| **統計 / Statistics** | `ret_5d_skew`, `ret_5d_kurt` | 報酬率偏度/峰度 / Return skewness/kurtosis |
| **統計 / Statistics** | `volatility_10d`, `volatility_20d` | 10日/20日波動率 / 10-day/20-day volatility |
| **市場 / Market** | `hsi_ret_5d`, `hsi_ret_20d` | 恒生指數漲跌幅 / Hang Seng Index return |
| **匯率 / FX** | `usdhkd_change` | 美元/港幣匯率變化 / USD/HKD exchange rate change |

**擴展特徵 (15 Features) / Extended Features (15 Features)：**

| 類別 / Category | 特徵 / Feature | 說明 / Description |
|---|---|---|
| **情緒分析 / Sentiment** | `sentiment_5d`, `sentiment_10d`, `sentiment_change` | 5日/10日情緒分數及變化 (新聞/社群情緒) / 5-day/10-day sentiment score and change (news/social) |
| **板塊輪動 / Sector** | `sector_momentum_5d`, `sector_momentum_20d`, `sector_vs_hsi` | 板塊動量及相對恒指表現 / Sector momentum and relative HSI performance |
| **沽空比率 / Short Selling** | `short_sell_ratio`, `short_sell_ratio_5d`, `short_sell_ratio_change` | 即時/5日沽空比率及變化 / Real-time/5-day short selling ratio and change |
| **互聯互通 / Connect Flow** | `southbound_net_5d`, `southbound_momentum`, `connect_sentiment` | 南向資金淨流入、動量、情緒 / Southbound net inflow, momentum, sentiment |
| **市場狀態 / Regime** | `market_regime`, `regime_confidence`, `hsi_trend_50_200` | 牛/熊/震盪狀態、信心度、均線比率 / Bull/Bear/Sideways regime, confidence, MA ratio |

### 模型訓練開關 / Model Training Toggles

| 環境變數 / Env Var | 預設值 / Default | 說明 / Description |
|---|---|---|
| `USE_ENSEMBLE` | `True` | 啟用模型集成 (False = 單模型比較) / Enable ensemble (False = single model comparison) |
| `USE_STACKING` | `False` | 使用 StackingClassifier (元模型學習組合) / Use StackingClassifier (meta-model learns combination) |
| `USE_BLENDING` | `False` | 使用 Blending (out-of-fold stacking，通常更準確) / Use Blending (out-of-fold stacking, usually more accurate) |
| `USE_CATBOOST` | `True` | 包含 CatBoost 作為第4個模型 / Include CatBoost as 4th model |
| `USE_SMOTE` | `True` | 啟用 SMOTE 類別不平衡處理 / Enable SMOTE class imbalance handling |
| `USE_GPU` | `False` | CatBoost 使用 GPU 訓練 (需要 NVIDIA GPU，會使用更多記憶體) / CatBoost GPU training (requires NVIDIA GPU, uses more memory) |

**優先級規則 / Priority Rules：**
- `USE_STACKING=True` 或 `USE_BLENDING=True` → 強制使用集成模式 / Forces ensemble mode
- `USE_ENSEMBLE=True` (預設) → VotingClassifier (加權平均) / VotingClassifier (weighted average)
- `USE_ENSEMBLE=False` → 單一最佳模型 (XGBoost vs LightGBM vs CatBoost) / Single best model

**訓練速度 / Training Speed：**
- 時間範圍 (1d, 5d, 20d) **平行訓練**，速度提升 ~3x / Timeframes (1d, 5d, 20d) **trained in parallel**, ~3x speedup
- 多支股票預測也支援**平行處理** / Multi-stock prediction also supports **parallel processing**

### 目標變數 (Target) / Target Variable
- **目標 / Target**: N天後收盤價 > 今日收盤價 → 1 (Buy)，否則 → 0 / N-day closing price > today's closing price → 1 (Buy), otherwise → 0
- **類別權重**: 自動平衡正負樣本 (上限3倍) + SMOTE 擴充 / **Class Weights**: Auto-balances positive/negative samples (max 3x) + SMOTE augmentation
- **多時間範圍 / Multi-timeframe**: 1天、5天、20天 / 1-day, 5-day, 20-day

### 模型表現 (F1 Score) / Model Performance (F1 Score)
| 時間範圍 / Timeframe | F1 Score | 說明 / Description |
|---|---|---|
| 1天 / 1-day | ~0.57 | 可用 — 短期趨勢 / Usable — short-term trend |
| 5天 / 5-day | ~0.69 | 良好 — 中期動量 / Good — mid-term momentum |
| 20天 / 20-day | ~0.73 | 最佳 — 長期趨勢 / Best — long-term trend |

**注意 / Note**: 股票預測本身非常困難，AUC ~0.55-0.60 已是合理範圍。 / Stock prediction is inherently difficult. An AUC of 0.55-0.60 is a reasonable range.

> **Note:** 股票預測本身非常困難。即使是大型對沖基金，AUC 也通常在 0.55-0.65 之間。AUC 0.55-0.60 代表模型有一定預測能力，但並非「錯誤」。 / Stock prediction is inherently difficult. Even large hedge funds typically achieve AUC between 0.55-0.65. An AUC of 0.55-0.60 indicates some predictive ability, not failure.

> **Warning:** 過去表現不代表未來結果。模型在歷史數據上的 F1/AUC 不保證未來收益。切勿將模型表現等同於交易保證。使用投資模擬器時，務必考慮交易成本和滑點的影響。 / Past performance does NOT guarantee future results. Model F1/AUC on historical data does not guarantee future returns. Never equate model performance with trading guarantees. When using the investment simulator, always consider trading costs and slippage impact.

### 信號判定 / Signal Determination
- 閾值由模型自動優化 (在驗證集上搜尋最佳 F1)，以下為預設值： / Thresholds are auto-optimized by the model (searches for best F1 on validation set). Defaults below:
| 信心度 / Confidence | 信號 / Signal |
|---|---|
| > 閾值 (優化後，預設55%) / > threshold (optimized, default 55%) | Buy (買入) |
| < 閾值 (優化後，預設45%) / < threshold (optimized, default 45%) | Sell (賣出) |
| 其餘 / Otherwise | Hold (持有) |

### 風險管理指標 / Risk Management Metrics

| 指標 / Metric | 說明 / Description | 計算方式 / Calculation |
|---|---|---|
| **預期報酬 / Expected Return** | 基於信心度和波動率估算 / Based on confidence & volatility | `(信心度-0.5) × 2 × 波動率 × √天數` |
| **止損點 / Stop Loss** | 建議止損位置 / Recommended stop loss level | `2 × 波動率 × √天數` |
| **止盈點 / Take Profit** | 建議止盈位置 / Recommended take profit level | `1.5 × \|預期報酬\|` |
| **風險報酬比 / Risk-Reward** | 收益與風險比例 / Gain to risk ratio | `報酬 / 風險` |
| **信心度趨勢 / Confidence Trend** | 信心度變化方向 / Direction of change | ↑上升 ↑down ↓下降 ↓down →持平 →flat |
| **勝率 / Win Rate** | 歷史預測準確率 / Historical accuracy | `Buy+Sell信號比例` / Buy+Sell signal ratio |

### 資料來源 / Data Sources
- **股票數據 / Stock Data**: yfinance (主) / akshare (備) / yfinance (primary) / akshare (fallback)
- **市場指數 / Market Index**: yfinance (^HSI 恒生指數 / Hang Seng Index)
- **匯率 / Exchange Rate**: yfinance (USD/HKD)

### 投資模擬 / Investment Simulation

根據歷史預測信號模擬跟單交易，計算實際投資收益。 / Simulates copy-trading based on historical prediction signals, calculates actual investment returns.

| 項目 / Item | 說明 / Description |
|---|---|
| **初始資金 / Initial Capital** | 每檔股票 HKD 20,000 (可在頁面自訂) / HKD 20,000 per stock (customizable on page) |
| **Buy 信號 / Buy Signal** | 以當日收盤價買入最大可購入股數 (整股) / Buy max whole shares at closing price |
| **Sell 信號 / Sell Signal** | 以當日收盤價賣出所有持股 / Sell all shares at closing price |
| **Hold 信號 / Hold Signal** | 不進行任何交易 (模擬結束時 counted as win/loss) / No trade (counted as win/loss at end) |
| **不做空 / No Short Selling** | Sell 信號僅用於平倉，不做空 / Sell only to close position, no shorting |

**交易成本 (香港標準) / Trading Costs (HK Standard)：**

| 費用 / Fee | 比例 / Rate | 說明 / Description |
|---|---|---|
| 佣金 / Commission | 0.1% | 每筆交易最低 HKD 20 / Minimum HKD 20 per trade |
| 印花稅 / Stamp Duty | 0.13% | 僅賣出時收取 / Sell-side only |

> **Note:** 交易成本按香港標準計算：佣金 0.1% (最低 HKD 20) + 印花稅 0.13% (僅賣出)。此費率可能因券商而異，實際成本請以你的券商為準。不同市場的交易成本差異很大，請勿直接套用於其他市場。 / Trading costs are based on HK standards: 0.1% commission (min HKD 20) + 0.13% stamp duty (sell-side only). Rates may vary by broker — verify with your broker. Trading costs vary greatly across markets; do not directly apply to other markets.

**計算公式 / Calculation Formulas：**
```
買入股數 = floor(可投資金額 / (股價 × (1 + 佣金率)))
Buy shares = floor(investable amount / (price × (1 + commission rate)))

賣出所得 = 股數 × 股價 × (1 - 佣金率 - 印花稅率)
Sell proceeds = shares × price × (1 - commission rate - stamp duty rate)

盈虧 = 賣出所得 - 買入成本
P/L = sell proceeds - buy cost
```

**輸出指標 / Output Metrics：**

| 指標 / Metric | 說明 / Description |
|---|---|
| 總盈虧 (HKD) / Total P/L | 投資組合最終價值 - 初始資金 / Final portfolio value - initial capital |
| 報酬率 (%) / Return Rate | (最終價值 - 初始資金) / 初始資金 × 100 / (Final - Initial) / Initial × 100 |
| 交易次數 / Trade Count | 買入 + 賣出次數 / Buy + sell count |
| 勝率 (%) / Win Rate | 盈利交易次數 / 總交易次數 × 100% (含持倉到期) / Winning trades / total trades × 100% (incl. positions held to expiry) |
| 最大回撤 (%) / Max Drawdown | 歷史最高點到最低點的跌幅百分比 / Percentage decline from peak to trough |
| Sharpe Ratio | 年化風險調整報酬率 (>1 不錯, >2 很好) / Annualized risk-adjusted return (>1 good, >2 great) |
| Sortino Ratio | 只考慮下跌風險的風險調整報酬率 / Risk-adjusted return considering only downside risk |
| 利潤因子 / Profit Factor | 總盈利 / 總虧損 (>1 表示盈利大於虧損) / Total profit / total loss (>1 means profit > loss) |
| 平均持倉天數 / Avg Holding Days | 每次買入到賣出的平均天數 / Average days from buy to sell |
| 買入持有報酬 / Buy & Hold Return | 基準策略：在開始時買入並持有到結束 / Baseline: buy at start and hold to end |
| 超額報酬 (Alpha) / Excess Return | 策略報酬 - 買入持有報酬 / Strategy return - buy & hold return |

**進階功能 / Advanced Features：**

| 功能 / Feature | 說明 / Description |
|---|---|
| **組合模擬 / Portfolio Simulation** | 模擬所有選中股票的組合表現，資金平均分配，顯示組合 Sharpe 和最大回撤 / Simulates portfolio of all selected stocks, equal capital allocation, shows portfolio Sharpe and max drawdown |
| **信心度加權 / Confidence-Weighted** | 根據信號信心度調整倉位大小 (30%-100% 資金)，高信心=大倉位 / Adjusts position size by signal confidence (30%-100% capital), high confidence = larger position |
| **蒙地卡羅測試 / Monte Carlo Test** | 隨機翻轉信號 1000 次，測試策略穩健性，顯示報酬分佈和獲利機率 / Randomly flips signals 1000 times, tests robustness, shows return distribution and profit probability |
| **預測準確度 / Prediction Accuracy** | 驗證歷史預測是否正確：Buy 後價格是否上漲？Sell 後價格是否下跌？ / Validates historical predictions: Did price rise after Buy? Fall after Sell? |
| **最佳時機模式 / Best Timing Mode** | 假設預知未來 N 天價格，選最佳買賣日 (後見之明模式) / Assumes foresight of N-day prices, picks best buy/sell days (hindsight mode) |

## 資料庫結構 / Database Schema

```sql
CREATE TABLE stock_predictions (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    stock_code TEXT NOT NULL,
    prediction_date DATE NOT NULL,
    timeframe TEXT CHECK (timeframe IN ('1d', '5d', '20d')),
    signal TEXT CHECK (signal IN ('Buy', 'Sell', 'Hold')),
    confidence FLOAT8,
    model_version TEXT,
    model_type TEXT,          -- 'voting', 'stacking', 'blending', 'xgboost', 'lightgbm', 'catboost'
    f1_score FLOAT8,          -- 模型 F1 分數 / Model F1 score
    auc_score FLOAT8,         -- 模型 AUC 分數 / Model AUC score
    expected_return FLOAT8,   -- 預期報酬率 (%) / Expected return (%)
    risk_reward FLOAT8,       -- 風險報酬比 / Risk-reward ratio
    stop_loss FLOAT8,         -- 止損點 (%) / Stop loss (%)
    take_profit FLOAT8,       -- 止盈點 (%) / Take profit (%)
    confidence_trend TEXT,    -- 信心度趨勢: ↑↓→- / Confidence trend
    win_rate FLOAT8,          -- 歷史勝率 (%) / Historical win rate (%)
    threshold_buy FLOAT8,     -- 優化後的 Buy 閾值 (每個時間範圍不同) / Optimized Buy threshold (varies per timeframe)
    threshold_sell FLOAT8,    -- 優化後的 Sell 閾值 (每個時間範圍不同) / Optimized Sell threshold (varies per timeframe)
    model_disagreement FLOAT8, -- 模型分歧度 (0=一致, 0.5=2v2, 1=完全分歧) / Model disagreement
    model_split TEXT,          -- 模型投票結果 (e.g., '4/0', '3/1', '2/2') / Model vote results
    created_at TIMESTAMPTZ DEFAULT NOW()
);
```

### 資料庫遷移 / Database Migrations

執行以下 SQL 語句來添加新欄位： / Execute the following SQL to add new columns:

```sql
-- 模型指標欄位 / Model metrics fields
ALTER TABLE stock_predictions ADD COLUMN IF NOT EXISTS model_type TEXT;
ALTER TABLE stock_predictions ADD COLUMN IF NOT EXISTS f1_score FLOAT8;
ALTER TABLE stock_predictions ADD COLUMN IF NOT EXISTS auc_score FLOAT8;

-- 風險管理欄位 / Risk management fields
ALTER TABLE stock_predictions ADD COLUMN IF NOT EXISTS expected_return FLOAT8;
ALTER TABLE stock_predictions ADD COLUMN IF NOT EXISTS risk_reward FLOAT8;
ALTER TABLE stock_predictions ADD COLUMN IF NOT EXISTS stop_loss FLOAT8;
ALTER TABLE stock_predictions ADD COLUMN IF NOT EXISTS take_profit FLOAT8;
ALTER TABLE stock_predictions ADD COLUMN IF NOT EXISTS confidence_trend TEXT DEFAULT '-';
ALTER TABLE stock_predictions ADD COLUMN IF NOT EXISTS win_rate FLOAT8;

-- 模型分歧指標 / Model disagreement metrics
ALTER TABLE stock_predictions ADD COLUMN IF NOT EXISTS model_disagreement FLOAT8 DEFAULT 0;
ALTER TABLE stock_predictions ADD COLUMN IF NOT EXISTS model_split TEXT DEFAULT '0/0';

-- 移除唯一限制 (保留歷史記錄) / Remove unique constraint (keep history)
ALTER TABLE stock_predictions DROP CONSTRAINT IF EXISTS unique_stock_prediction;
```

## 模型監控 / Model Monitoring

### 數據品質檢查 / Data Quality Checks
- 檢測缺失日期 (每個時間範圍至少需要 5 筆預測) / Detect missing dates (minimum 5 predictions per timeframe)
- 檢查信心度分佈是否合理性 / Check confidence distribution is reasonable
- 檢測信號分佈是否異常 (某信號 >70%) / Detect abnormal signal distribution (single signal >70%)

### 真實準確度驗證 / True Accuracy Verification
- 透過 yfinance 獲取實際價格數據 / Fetch actual price data via yfinance
- Buy 信號：N天後收盤價 > 預測日收盤價 = 正確 / Buy signal: N-day closing > prediction day closing = correct
- Sell 信號：N天後收盤價 < 預測日收盤價 = 正確 / Sell signal: N-day closing < prediction day closing = correct
- 顯示各股票、各時間範圍的真實準確度百分比 / Display true accuracy percentage per stock and timeframe

### 模型漂移檢測 / Model Drift Detection
- 比較最近 14 天 vs 60 天的預測準確度 (使用真實價格驗證) / Compare recent 14-day vs 60-day prediction accuracy (verified with actual prices)
- 準確度下降 >10% = 高度警報 / Accuracy drop >10% = high alert
- 準確度下降 >5% = 中度警報 / Accuracy drop >5% = moderate alert

### 回測引擎 / Backtest Engine
- 實際模擬策略表現：買入 → 賣出 → 計算真實回報 / Actually simulates strategy: buy → sell → calculate real returns
- 計算 Sharpe Ratio、勝率、最大回撤 / Calculates Sharpe Ratio, win rate, max drawdown
- 對比買入持有基準，計算超額報酬 (Alpha) / Compares to buy & hold baseline, calculates excess return (Alpha)

### 信號警報 / Signal Alerts
- 信心度 >70% 的強勢信號 / Strong signals with confidence >70%
- 預期報酬 >5% 的高回報信號 / High-return signals with expected return >5%

### 信心度校準 / Confidence Calibration
- 監控平均信心度是否合理性 / Monitors whether average confidence is reasonable
- 過度自信 (>60%) 或信心不足 (<40%) 會建議調整 / Over-confidence (>60%) or under-confidence (<40%) triggers adjustment suggestions

## 注意事項 / Important Notes

- 所有日期時間使用香港時區 (`Asia/Hong_Kong`) / All dates/times use Hong Kong timezone (`Asia/Hong_Kong`)
- 特徵計算嚴禁 Look-ahead Bias（只使用當天之前的數據） / Feature computation MUST NOT contain Look-ahead Bias (only use data before the current day)
- `.env` 檔案包含敏感資訊，請勿上傳至版本控制 / `.env` contains sensitive info — do not upload to version control
- 模型檔案 (`.pkl`) 不上傳至版本控制 / Model files (`.pkl`) must not be uploaded to version control
- 每日預測會保留歷史記錄 (自動清理 60 天前的舊數據) / Daily predictions keep history (auto-clean data older than 60 days)

> **Warning:** 特徵工程嚴禁 Look-ahead Bias。這是量化交易中最常見的錯誤之一。所有特徵必須僅使用當天及之前的数据計算，確保模型在實際交易中能複製相同的表現。 / Feature engineering MUST NOT contain Look-ahead Bias. This is one of the most common mistakes in quantitative trading. All features must be computed using only data from the current day and earlier, ensuring the model can replicate the same performance in live trading.

> **Warning:** `.env` 檔案包含 Supabase 金鑰和資料庫憑證。絕對不要提交至 Git，也不要出現在 Docker 鏡像中。建議在 `.gitignore` 中明確排除 `.env`、`*.pkl` 和 `models/` 目錄。 / `.env` files contain Supabase keys and database credentials. NEVER commit to Git, and never include in Docker images. Explicitly exclude `.env`, `*.pkl`, and `models/` in `.gitignore`.

## 常見問題 / FAQ

### Q: 為什麼 AUC 只有 0.55-0.60？ / Why is AUC only 0.55-0.60?
A: 股票預測本身非常困難。即使是大型對沖基金，AUC 也通常在 0.55-0.65 之間。你的模型已達到合理範圍。 / Stock prediction is inherently difficult. Even large hedge funds typically achieve AUC between 0.55-0.65. Your model is within a reasonable range.

> **Note:** 股票預測不是有明確答案的問題。市場充滿噪音和隨機性，AUC 0.55-0.60 已代表具有統計顯著性的預測能力。 / Stock prediction is not a problem with clear answers. Markets are full of noise and randomness. AUC 0.55-0.60 already represents statistically significant predictive ability.

### Q: F1 Score 代表什麼？ / What does F1 Score mean?
A: F1 = 精準率與召回率的平衡。F1 > 0.5 表示模型比隨機好，F1 > 0.6 表示可用於交易信號。 / F1 = balance between precision and recall. F1 > 0.5 means model is better than random, F1 > 0.6 means usable for trading signals.

### Q: 什麼是模型集成 (Ensemble)？ / What is Model Ensemble?
A: 同時訓練 XGBoost、LightGBM、RandomForest、CatBoost 四個模型，透過 VotingClassifier (加權平均)、StackingClassifier (元模型學習) 或 Blending (out-of-fold stacking) 結合它們的預測機率。通常比單一模型更穩定、AUC 更高。 / Trains XGBoost, LightGBM, RandomForest, CatBoost simultaneously, combines their prediction probabilities via VotingClassifier (weighted average), StackingClassifier (meta-model learning), or Blending (out-of-fold stacking). Usually more stable and higher AUC than single models.

### Q: Voting、Stacking、Blending 有什麼差別？ / What's the difference between Voting, Stacking, Blending?
A: 
- **Voting**: 用加權平均結合四個模型的預測機率 (預設，最快) / Combines four models' probabilities via weighted average (default, fastest)
- **Stacking**: 用一個元模型 (LogisticRegression) 學習如何最佳組合四個模型的預測 (較慢但通常更準) / Uses a meta-model (LogisticRegression) to learn optimal combination (slower but usually more accurate)
- **Blending**: 類似 Stacking，但使用 out-of-fold predictions 避免過擬合 (最慢但通常最準) / Similar to Stacking, but uses out-of-fold predictions to avoid overfitting (slowest but usually most accurate)

### Q: CatBoost 是什麼？為什麼要加它？ / What is CatBoost? Why add it?
A: CatBoost 是 Yandex 開發的梯度提升框架，對類別型特徵處理更好，在金融數據上通常表現優於 XGBoost/LightGBM。加入後可提升集成模型的準確度。 / CatBoost is a gradient boosting framework by Yandex, handles categorical features better, usually outperforms XGBoost/LightGBM on financial data. Adding it improves ensemble accuracy.

### Q: SMOTE 是什麼？為什麼需要它？ / What is SMOTE? Why is it needed?
A: SMOTE (Synthetic Minority Over-sampling Technique) 在訓練集上生成少數類的合成樣本，解決正負樣本不平衡問題。僅在 TimeSeriesSplit 的訓練折上套用，不會洩漏未來資訊。 / SMOTE (Synthetic Minority Over-sampling Technique) generates synthetic samples for minority classes on the training set to solve class imbalance. Applied only on training folds of TimeSeriesSplit — no future data leakage.

> **Warning:** SMOTE 僅在訓練折上套用，絕對不能在驗證折或測試集上套用。 / SMOTE must only be applied on training folds, NEVER on validation folds or test sets.

### Q: 可以添加更多股票嗎？ / Can I add more stocks?
A: 修改 `.env` 中的 `STOCK_LIST`，例如 `STOCK_LIST=0700,9988,0005,0939,1810` / Edit `STOCK_LIST` in `.env`, e.g. `STOCK_LIST=0700,9988,0005,0939,1810`

### Q: 如何查看訓練日誌？ / How to view training logs?
A: 日誌位於 `logs/app.log` / Logs are at `logs/app.log`

### Q: 訓練要多久？ / How long does training take?
A: 約 3-5 分鐘 (使用平行訓練，時間範圍同時訓練)。若使用 Stacking 或 Blending，約 5-8 分鐘。 / About 3-5 minutes (parallel training across timeframes). If using Stacking or Blending, about 5-8 minutes.

### Q: 預測要多久？ / How long does prediction take?
A: 平行預測多支股票，約 5-10 秒 (取決於股票數量)。 / Parallel prediction across multiple stocks, about 5-10 seconds (depends on stock count).

### Q: 什麼是特徵相關性過濾？ / What is Feature Correlation Filter?
A: 訓練前自動移除 |corr| > 0.9 的冗餘特徵。例如 `ret_3d` 與 `ret_1d`/`ret_5d` 高度相關，只保留最具資訊量的一個。減少噪音、加快訓練、降低過擬合。 / Auto-removes redundant features with |corr| > 0.9 before training. E.g., `ret_3d` is highly correlated with `ret_1d`/`ret_5d` — only the most informative is kept. Reduces noise, speeds up training, lowers overfitting.

### Q: 閾值優化是什麼？ / What is Threshold Optimization?
A: 一般系統用固定閾值 (Buy > 55%, Sell < 45%)，但不同時間範圍的最佳閾值不同。系統會在驗證集上自動搜尋使 F1 最高的 Buy/Sell 閾值，訓練後存入模型。 / Typical systems use fixed thresholds (Buy > 55%, Sell < 45%), but optimal thresholds differ per timeframe. The system auto-searches for Buy/Sell thresholds that maximize F1 on validation, saves them after training.

### Q: 模型版本化有什麼用？ / What is Model Versioning?
A: 每次訓練會保存帶時間戳的模型備份 (如 `best_model_5d_20260822_163000.pkl`)，自動保留最近 5 版。如果新模型效果不好，可以手動回滾到舊版。 / Each training saves a timestamped model backup (e.g. `best_model_5d_20260822_163000.pkl`), keeps latest 5. If new model performs poorly, manually rollback.

### Q: 特徵重要性 CSV 怎麼用？ / How to use Feature Importance CSV?
A: 訓練後自動輸出至 `models/feature_importance_{timeframe}.csv`。可用 Excel 開啟分析哪些特徵對模型預測最有影響，協助特徵工程優化。 / Auto-output after training to `models/feature_importance_{timeframe}.csv`. Open in Excel to analyze which features most influence model predictions, helping feature engineering optimization.

### Q: 模型會自動更新嗎？ / Does the model auto-update?
A: 需要手動執行 `train_model.py` 重新訓練，或設定 Windows 排程器自動執行 / Requires manual `train_model.py` execution or setting up Windows Task Scheduler

### Q: 如何解讀止損/止盈點？ / How to interpret Stop Loss/Take Profit?
A: 
- Buy 信號：止損為負數 (下跌止損)，止盈為正數 (上漲獲利) / Buy signal: Stop loss is negative (downside stop), take profit is positive (upside gain)
- Sell 信號：止損為正數 (上漲止損)，止盈為負數 (下跌獲利) / Sell signal: Stop loss is positive (upside stop), take profit is negative (downside gain)

### Q: 模型漂移是什麼？ / What is Model Drift?
A: 模型漂移是指模型預測能力隨時間下降。系統會自動檢測並提醒您重新訓練。 / Model drift means model prediction capability degrades over time. System auto-detects and prompts retraining.

### Q: 信心度校準有什麼用？ / What is Confidence Calibration?
A: 確保信心度分數可靠。如果模型過度自信或信心不足，系統會建議調整。 / Ensures confidence scores are reliable. If model is over-confident or under-confident, system suggests adjustments.

### Q: 投資模擬器是什麼？ / What is the Investment Simulator?
A: 投資模擬器根據歷史預測信號 (Buy/Sell/Hold) 模擬跟單交易。每檔股票以 HKD 20,000 初始資金，Buy 時買入、Sell 時賣出，計算實際收益、勝率和最大回撤。可在 Streamlit 側邊欄切換至「💰 投資模擬器」頁面使用。 / Investment simulator mimics copy-trading based on historical signals (Buy/Sell/Hold). HKD 20,000 initial capital per stock, buys on Buy, sells on Sell, calculates actual returns, win rate, max drawdown. Switch to "💰 Investment Simulator" in the Streamlit sidebar.

### Q: 模擬結果包含交易成本嗎？ / Do simulation results include trading costs?
A: 是的。模擬包含香港標準交易成本：佣金 0.1% (每筆最低 HKD 20) + 印花稅 0.13% (僅賣出)。計算出的盈虧為扣除成本後的淨收益。 / Yes. Simulation includes HK standard trading costs: 0.1% commission (min HKD 20) + 0.13% stamp duty (sell-side only). P/L is net of costs.

### Q: 模擬器可以做空嗎？ / Can the simulator short-sell?
A: 不可以。Sell 信號僅用於賣出已持有的股票 (平倉)，不進行做空操作。若沒有持股時出現 Sell 信號，則不執行任何交易。 / No. Sell signals only close existing positions — no short selling. If a Sell signal appears with no holdings, no trade is executed.

### Q: 什麼是 Sharpe Ratio？ / What is Sharpe Ratio?
A: Sharpe Ratio 是年化風險調整報酬率。計算方式為 (平均報酬 - 無風險利率) / 報酬標準差 × √252。一般來說：>1 表示不錯，>2 表示很好，<0 表示不如無風險投資。 / Sharpe Ratio is annualized risk-adjusted return. Calculated as (mean return - risk-free rate) / return std × √252. Generally: >1 is good, >2 is great, <0 means worse than risk-free investment.

### Q: 什麼是 Sortino Ratio？ / What is Sortino Ratio?
A: Sortino Ratio 與 Sharpe Ratio 類似，但只考慮下跌風險 (負報酬的標準差)。適合評估不對稱報酬的投資策略。 / Similar to Sharpe Ratio, but only considers downside risk (std of negative returns). Suitable for evaluating asymmetric return strategies.

### Q: 什麼是利潤因子？ / What is Profit Factor?
A: 利潤因子 = 總盈利 / 總虧損。>1 表示盈利大於虧損，>2 表示很好。如果利潤因子 < 1，表示策略整體虧損。 / Profit Factor = total profit / total loss. >1 means profit exceeds loss, >2 is great. If < 1, the strategy is overall losing.

### Q: 什麼是買入持有基準？ / What is Buy & Hold Benchmark?
A: 買入持有 (Buy & Hold) 是最簡單的投資策略：在開始時買入並持有到結束。系統會將信號策略的報酬與買入持有對比，顯示超額報酬 (Alpha)。正 Alpha 表示策略跑贏大盤。 / Buy & Hold is the simplest strategy: buy at start and hold to end. System compares signal strategy return to buy & hold, showing excess return (Alpha). Positive Alpha means strategy outperforms the market.

### Q: 什麼是組合模擬？ / What is Portfolio Simulation?
A: 組合模擬將資金平均分配到所有選中的股票，模擬多股票組合的表現。這可以顯示分散投資的效果，以及組合的整體 Sharpe Ratio 和最大回撤。 / Portfolio simulation allocates capital equally across all selected stocks, simulates multi-stock portfolio performance. Shows diversification effect, portfolio Sharpe Ratio and max drawdown.

### Q: 什麼是信心度加權策略？ / What is Confidence-Weighted Strategy?
A: 信心度加權策略根據信號的信心度調整倉位大小。高信心信號 (如 90%) 會投入更多資金 (最多 100%)，低信心信號 (如 50%) 只投入較少資金 (最少 30%)。這可以讓你在更有把握的交易上投入更多。 / Confidence-weighted strategy adjusts position size by signal confidence. High confidence signals (e.g., 90%) invest more capital (up to 100%), low confidence signals (e.g., 50%) invest less (min 30%). This lets you invest more in higher-conviction trades.

### Q: 什麼是蒙地卡羅測試？ / What is Monte Carlo Test?
A: 蒙地卡羅測試隨機翻轉信號 (如 20% 的 Buy/Sell 信號被隨機交換)，然後運行 1000 次模擬。這可以測試策略的穩健性：如果策略在信號被隨機干擾後仍然盈利，表示策略較為可靠。 / Monte Carlo test randomly flips signals (e.g., 20% of Buy/Sell signals randomly swapped), runs 1000 simulations. Tests strategy robustness: if strategy remains profitable after random signal disruption, it's more reliable.

### Q: 預測準確度如何計算？ / How is Prediction Accuracy Calculated?
A: 系統會驗證歷史預測是否正確：Buy 信號 → N天後收盤價是否上漲？Sell 信號 → N天後收盤價是否下跌？準確度 = 正確預測數 / 總預測數 × 100%。這使用真實價格數據驗證，而非僅用信心度作為代理。 / System validates historical predictions: Buy signal → did N-day closing price rise? Sell signal → did it fall? Accuracy = correct predictions / total predictions × 100%. Uses actual price data, not just confidence as proxy.

### Q: 勝率和預測準確度有什麼差別？ / What's the difference between Win Rate and Prediction Accuracy?
A: 勝率是指模擬交易中盈利的交易比例 (賣出或持倉到期時計算)。預測準確度是指信號方向是否正確 (價格是否朝預測方向移動)。兩者可能不同，因為勝率還受到交易成本、進出場時機等因素影響。 / Win rate is the proportion of profitable trades in the simulation (calculated at sell or expiry). Prediction accuracy is whether the signal direction was correct (price moved as predicted). They can differ because win rate is also affected by trading costs, entry/exit timing, etc.

### Q: 什麼是模型分歧 (Model Disagreement)？ / What is Model Disagreement?
A: 模型分歧是指四個模型 (XGBoost, LightGBM, RandomForest, CatBoost) 預測結果不一致的程度。例如： / Model disagreement is the degree of inconsistency among the four models (XGBoost, LightGBM, RandomForest, CatBoost). For example:
- `4/0`：四個模型都認為 Buy → 分歧度 0 (完全一致) / All four say Buy → disagreement 0 (full agreement)
- `3/1`：三個 Buy，一個 Sell → 分歧度 0.25 / Three Buy, one Sell → disagreement 0.25
- `2/2`：兩個 Buy，兩個 Sell → 分歧度 0.5 (五五波) / Two Buy, two Sell → disagreement 0.5 (split)
系統會在分歧度 >= 50% 時強制將信號設為 Hold，避免在模型意見分歧時做出弱勢決策。 / System forces Hold when disagreement >= 50% to avoid weak decisions when models disagree.

### Q: 模型分歧時為什麼要強制 Hold？ / Why force Hold on Model Disagreement?
A: 當模型意見分歧時 (如 2v2)，表示市場方向不明確。此時做出 Buy 或 Sell 決策風險較高。強制 Hold 可以避免在不確定時刻進場，保護資金安全。儀表板會顯示 ⚠️ 警告標示分歧程度。 / When models disagree (e.g., 2v2), market direction is unclear. Buy/Sell decisions carry higher risk. Forced Hold avoids entering during uncertainty, protecting capital. Dashboard shows ⚠️ warning indicating disagreement level.

### Q: K線圖上的買賣信號怎麼看？ / How to read Buy/Sell signals on K-line chart?
A: K線圖上標示了模型預測的買入 (▲ 綠色) 和賣出 (▼ 紅色) 信號位置。三角形位置對應預測日的價格，您可以直觀看到：信號發出後價格是否朝預測方向移動。例如 Buy 信號後價格上漲，表示預測正確。 / K-line chart marks model-predicted buy (▲ green) and sell (▼ red) signal positions. Triangle position corresponds to prediction day price — you can visually see if price moved in the predicted direction after the signal. E.g., price rising after Buy means correct prediction.

### Q: 技術指標的作用是什麼？ / What is the role of Technical Indicators?
A: 技術指標 (RSI、MACD、Stochastic 等) 是模型學習的輸入特徵。儀表板顯示這些指標的最新數值，讓您了解：模型做出 Buy/Sell 決策時，技術面是否支持這個判斷。例如 RSI > 70 (超買) 時出現 Buy 信號，可能需要謹慎。 / Technical indicators (RSI, MACD, Stochastic, etc.) are model input features. Dashboard shows latest values, helping you understand if technicals support the model's Buy/Sell decision. E.g., a Buy signal when RSI > 70 (overbought) warrants caution.

### Q: 信號確認分析 (Signal Confirmation) 是什麼？ / What is Signal Confirmation Analysis?
A: 每個 Buy/Sell 信號下方會自動分析當前指標是否支持該信號。例如： / Each Buy/Sell signal auto-analyzes whether current indicators support it. For example:
- ✅ RSI 42 偏低 → 支持 Buy / RSI 42 low → supports Buy
- ❌ MACD -1.06 空頭動能 → 矛盾 Buy / MACD -1.06 bearish momentum → contradicts Buy
- ➖ ADX 9 盤整 → 信號可靠性降低 / ADX 9 ranging → signal reliability reduced
最後顯示「幾項支持、幾項中性、幾項矛盾」，幫助快速判斷信號可靠性。Hold 信號不會顯示分析。 / Finally shows "X support, X neutral, X contradict" to quickly assess signal reliability. Hold signals do not display analysis.

### Q: 技術指標是即時計算的嗎？ / Are Technical Indicators calculated in real-time?
A: 是的。指標從 yfinance 即時獲取 2 年 OHLCV 數據，透過 `compute_features()` 計算。結果緩存 5 分鐘 (TTL=300s)，避免重複計算。如果今日市場尚未收盤，今日數據的 Close 可能為 NaN，系統會自動跳過該筆數據。 / Yes. Indicators fetch 2 years of OHLCV data from yfinance in real-time, computed via `compute_features()`. Results cached 5 minutes (TTL=300s) to avoid recomputation. If market hasn't closed today, today's Close may be NaN — system auto-skips.

### Q: 滾動準確度怎麼計算？ / How is Rolling Accuracy calculated?
A: 滾動準確度 = 最近 30 天內正確預測數 / 總預測數 × 100%。驗證方式：Buy 信號 → N天後收盤價是否上漲？Sell 信號 → N天後收盤價是否下跌？這使用真實價格數據，比單看訓練指標 (F1/AUC) 更可靠。 / Rolling accuracy = correct predictions in last 30 days / total predictions × 100%. Verified by: Buy signal → did N-day closing rise? Sell signal → did it fall? Uses actual price data, more reliable than training metrics (F1/AUC) alone.

### Q: 如何執行測試？ / How to run tests?
A: 使用 pytest 執行測試：`python -m pytest tests/ -v`。測試覆蓋環境變數設定、特徵工程、模型訓練、預測上傳等核心功能。 / Run tests with pytest: `python -m pytest tests/ -v`. Tests cover env config, feature engineering, model training, prediction upload core functionality.

### Q: 測試覆蓋了哪些功能？ / What features are covered by tests?
A: 共 85 個測試，涵蓋： / 85 tests covering:
- 環境變數載入與驗證 (10 個) / Env var loading & validation (10)
- 技術指標計算：RSI、MACD、ATR、ADX、Stochastic、MFI、Williams %R (17 個) / Technical indicator calculation: RSI, MACD, ATR, ADX, Stochastic, MFI, Williams %R (17)
- 模型訓練：XGBoost、LightGBM、RandomForest、CatBoost (含 early stopping)、SMOTE、Blending (14 個) / Model training: XGBoost, LightGBM, RandomForest, CatBoost (incl. early stopping), SMOTE, Blending (14)
- 預測功能：日期計算、模型載入、信號判定、上傳 (10 個) / Prediction: date calculation, model loading, signal determination, upload (10)
- 擴展特徵：情緒、板塊、沽空、互聯互通、市場狀態、增量學習、動態權重 (24 個) / Extended features: sentiment, sector, short selling, connect flow, regime, online learning, dynamic weighting (24)
- 儀表板頁面：回測頁面、投資組合頁面 (4 個) / Dashboard pages: backtest page, portfolio page (4)

### Q: 15 項擴展特徵是什麼？ / What are the 15 Extended Features?
A: 擴展特徵分為兩階段新增，從外部數據源獲取額外資訊： / Extended features added in two phases, fetching additional data from external sources:

**Phase 1 (10 Features) / 第一階段：**
- **情緒分析 (3)**: 從新聞/社群獲取市場情緒分數 / **Sentiment (3)**: Market sentiment score from news/social media
- **板塊輪動 (3)**: 追蹤行業板塊ETF動量，判斷資金輪動方向 / **Sector Rotation (3)**: Tracks sector ETF momentum for fund rotation
- **沽空比率 (3)**: 監控沽空活動變化，高沽空可能暗示看跌情緒 / **Short Selling (3)**: Monitors short selling activity — high shorting may signal bearish sentiment
- **互聯互通 (1)**: 南向資金流向反映內地投資者情緒 / **Connect Flow (1)**: Southbound capital flow reflects mainland investor sentiment

**Phase 2 (5 Features) / 第二階段：**
- **市場狀態 (3)**: 使用恒指 MA50/MA200 交叉判斷牛市/熊市/震盪 / **Regime (3)**: Uses HSI MA50/MA200 crossover for bull/bear/sideways detection

所有擴展特徵在無法獲取數據時自動回退為預設值，不影響核心功能。 / All extended features fall back to defaults when data is unavailable — core functionality is unaffected.

### Q: 市場狀態偵測如何影響預測？ / How does Regime Detection affect predictions?
A: 市場狀態分為三種：牛市 (MA50>MA200)、熊市 (MA50<MA200)、震盪 (信號混合)。牛市中 Buy 信號更可靠，熊市中 Sell 信號更可靠。儀表板會顯示當前狀態 (🟢/🔴/🟡)，供參考。 / Regime has three states: Bull (MA50>MA200), Bear (MA50<MA200), Sideways (mixed). Buy signals are more reliable in Bull markets, Sell signals in Bear. Dashboard displays current state (🟢/🔴/🟡) for reference.

### Q: 增量學習是什麼？ / What is Online Learning?
A: 增量學習允許模型在不完全重訓的情況下，用近期數據進行增量更新。當上次全量訓練超過 7 天時，系統會自動使用最近 60 天數據更新模型權重，節省時間和計算資源。目前預設關閉 (USE_ONLINE_LEARNING=False)。 / Online learning allows incremental model updates using recent data without full retraining. When last full train was >7 days ago, system auto-updates with last 60 days of data. Currently disabled by default (USE_ONLINE_LEARNING=False).

### Q: 動態權重如何運作？ / How does Dynamic Weighting work?
A: 動態權重追蹤每個模型 (XGBoost/LightGBM/RandomForest/CatBoost) 的近期表現，使用 EMA (指數移動平均) 調整集成權重。表現較好的模型獲得較高權重，使集成預測更準確。目前預設關閉 (USE_DYNAMIC_WEIGHTING=False)。 / Dynamic weighting tracks each model's recent performance, uses EMA to adjust ensemble weights. Better-performing models get higher weights. Currently disabled by default (USE_DYNAMIC_WEIGHTING=False).

## 近期更新 / Recent Updates

### 新增功能 (2026-09-15) / New Features (2026-09-15)

#### Phase 1: Data Features — 10 New Features

| Feature | Module | Description |
|---|---|---|
| `sentiment_5d` | `src/sentiment.py` | 5-day rolling news sentiment score from AKShare / 5天滾動新聞情緒分數 |
| `sentiment_10d` | `src/sentiment.py` | 10-day rolling news sentiment score / 10天滾動新聞情緒分數 |
| `sentiment_change` | `src/sentiment.py` | Change in sentiment (5d vs 10d) / 情緒變化 (5天 vs 10天) |
| `sector_momentum_5d` | `src/sector.py` | 5-day sector ETF momentum (XLF, XLK, XLE) / 5天板塊ETF動量 |
| `sector_momentum_20d` | `src/sector.py` | 20-day sector ETF momentum / 20天板塊ETF動量 |
| `sector_vs_hsi` | `src/sector.py` | Sector performance relative to HSI / 板塊相對恒指表現 |
| `short_sell_ratio` | `src/short_selling.py` | Estimated short selling ratio / 估算沽空比率 |
| `short_sell_ratio_5d` | `src/short_selling.py` | 5-day rolling short sell ratio / 5天滾動沽空比率 |
| `short_sell_ratio_change` | `src/short_selling.py` | Change in short sell ratio / 沽空比率變化 |
| `southbound_net_5d` | `src/connect_flow.py` | 5-day net southbound flow / 5天淨南向資金流 |
| `southbound_momentum` | `src/connect_flow.py` | Southbound flow momentum / 南向資金動量 |
| `connect_sentiment` | `src/connect_flow.py` | Connect market sentiment / 互聯互通市場情緒 |

#### Phase 2: Model Features — 3 New Features

| Feature | Module | Description |
|---|---|---|
| `market_regime` | `src/regime.py` | Market regime: 0=bear, 1=sideways, 2=bull / 市場狀態：0=熊市, 1=震盪, 2=牛市 |
| `regime_confidence` | `src/regime.py` | Confidence of regime classification (0-1) / 狀態分類信心度 |
| `hsi_trend_50_200` | `src/regime.py` | HSI 50-day MA vs 200-day MA ratio / 恒指50日均線 vs 200日均線比率 |

#### Phase 3: Dashboard — 2 New Pages

| Page | File | Description |
|---|---|---|
| Backtest | `app/pages/2_backtest.py` | Equity curve, drawdown, trade log, vs buy-and-hold benchmark / 資金曲線、回撤、交易記錄、對比買入持有 |
| Portfolio | `app/pages/3_portfolio.py` | Holdings summary, signal/confidence distribution, risk exposure / 持倉總覽、信號/信心度分佈、風險暴露 |

#### Model Improvements

| Feature | File | Description |
|---|---|---|
| Online Learning | `src/online_learner.py` | Incremental model updates with warm-start XGBoost/LightGBM / 增量學習，熱啟動更新模型 |
| Dynamic Weighting | `src/dynamic_weighting.py` | EMA-based ensemble weight adjustment by recent performance / 基於近期表現的動態集成權重調整 |
| Market Regime Display | `app/streamlit_app.py` | Regime indicator in signal cards and performance tab / 信號卡片和模型表現分頁顯示市場狀態 |

#### New Environment Variables

```env
# Phase 1
USE_SENTIMENT=True     # News sentiment features
USE_SECTOR=True        # Sector rotation features
USE_SHORT_SELL=True    # Short selling features
USE_CONNECT=True       # Northbound/Southbound flow features

# Phase 2
USE_ONLINE_LEARNING=False   # Incremental model updates
USE_REGIME=True             # Market regime detection
USE_DYNAMIC_WEIGHTING=False # Dynamic ensemble weights
```

**No database schema changes required** — All 15 new features are computed in-memory during training and prediction, and are NOT stored in the database. The `stock_predictions` table schema remains unchanged.

**新增 8 個測試檔案** — 34 個新測試，全部 85 個測試通過。 / **8 new test files** — 34 new tests, all 85 tests passing.

### 改進項目 (2026-09-12) / Improvements (2026-09-12)

| 改進 / Improvement | 說明 / Description |
|---|---|
| **分頁式儀表板 / Tabbed Dashboard** | 預測儀表板重構為 5 個分頁：信號總覽、K線與指標、信心度趨勢、預測記錄、模型表現 / Dashboard restructured into 5 tabs: Signal Overview, K-line & Indicators, Confidence Trend, Prediction Records, Model Performance |

### 改進項目 (2026-09-11) / Improvements (2026-09-11)

| 改進 / Improvement | 說明 / Description |
|---|---|
| **K線圖 (含買賣信號) / K-line Chart** | 新增互動式K線圖，標示模型預測的買入(▲)/賣出(▼)信號位置，支援自選股票和時間範圍 (30/60/90/180天) / Interactive K-line chart with model-predicted buy(▲)/sell(▼) signals, supports custom stock and timeframe (30/60/90/180 days) |
| **技術指標展示 / Technical Indicators** | 新增技術指標面板，顯示 RSI、MACD、Stochastic、ADX、MFI、布林帶寬、ATR、量比等最新數值，作為信號依據 / Technical indicator panel showing RSI, MACD, Stochastic, ADX, MFI, Bollinger Band Width, ATR, Volume Ratio as signal basis |
| **信號確認分析 / Signal Confirmation** | 每個 Buy/Sell 信號下方自動分析指標 alignment，顯示「幾項支持/矛盾/中性」及每個指標的具體判斷 / Auto-analyzes indicator alignment below each Buy/Sell signal, shows "X support/contradict/neutral" with per-indicator assessment |
| **滾動準確度追蹤 / Rolling Accuracy** | 新增30天滾動窗口計算真實預測準確度，驗證 Buy 後價格是否上漲、Sell 後價格是否下跌 / 30-day rolling window for true prediction accuracy, verifies price direction after Buy/Sell |
| **訓練指標展示 / Training Metrics** | 新增訓練指標面板，顯示各股票各時間範圍的 F1 Score、AUC Score、冠軍模型類型 / Training metrics panel showing F1, AUC, champion model per stock and timeframe |
| **Bug Fix: MFI 缺失 / MFI Missing** | 修正 `get_latest_indicators()` 回傳字典缺少 `mfi` 鍵，導致 MFI 永遠顯示 `—` / Fixed `get_latest_indicators()` missing `mfi` key causing MFI to always display `—` |
| **Bug Fix: MA50 偏離 / MA50 Deviation** | 修正 MA50 偏離顯示錯誤 (原始值 -0.05% → 正確值 -4.85%)，因小數未乘以 100 / Fixed MA50 deviation display (raw -0.05% → correct -4.85%) due to missing ×100 |
| **Bug Fix: 準確度顯示 / Accuracy Display** | 修正滾動準確度顯示 5000.0% (因 `:.1%` 格式誤將百分比當小數處理) / Fixed rolling accuracy showing 5000.0% (`:.1%` format treated percentage as decimal) |
| **Bug Fix: 鍵名錯誤 / Wrong Key Name** | 修正 `total_predictions`/`correct_predictions` → `total`/`correct` (與 `calculate_accuracy` 回傳值一致) / Fixed `total_predictions`/`correct_predictions` → `total`/`correct` (matches `calculate_accuracy` return value) |
| **NaN 資料處理 / NaN Handling** | 修正 yfinance 今日數據 Close 為 NaN 導致所有指標顯示 `—`，改為先 `dropna(subset=["Close"])` / Fixed yfinance today's Close NaN causing all indicators to show `—`, now `dropna(subset=["Close"])` first |

**無資料庫變更** — 所有功能讀取現有 `stock_predictions` 表或即時計算。 / **No DB changes** — All features read existing `stock_predictions` table or compute in real-time.

### 修改的檔案 / Modified Files
- `app/streamlit_app.py` — 分頁式儀表板重構；新增 K線圖、技術指標、信號確認分析、滾動準確度、訓練指標展示；修正 MFI、MA50、準確度顯示 bug / Tabbed dashboard restructure; added K-line, technical indicators, signal confirmation, rolling accuracy, training metrics; fixed MFI, MA50, accuracy display bugs

### 改進項目 (2026-09-10) / Improvements (2026-09-10)

| 改進 / Improvement | 說明 / Description |
|---|---|
| **CatBoost 模型 / CatBoost Model** | 新增 CatBoost 作為第4個模型選項，通常在金融數據上表現更好 / Added CatBoost as 4th model option, usually better on financial data |
| **Blending 集成 / Blending** | 新增 Blending 模式 (out-of-fold stacking)，通常比 Voting 更準確 / Added Blending mode (out-of-fold stacking), usually more accurate than Voting |
| **平行訓練 / Parallel Training** | 時間範圍 (1d, 5d, 20d) 平行訓練，速度提升 ~3x / Timeframes (1d, 5d, 20d) trained in parallel, ~3x speedup |
| **平行預測 / Parallel Prediction** | 多支股票預測平行處理，大幅提升預測速度 / Multi-stock prediction parallel processing, significant speed improvement |
| **模型比較表 / Model Comparison** | 訓練時顯示各模型 F1 分數比較，清楚標示贏家 / Training shows F1 comparison across models, clearly marks winner |
| **模型分歧檢測 / Disagreement Detection** | 當模型意見分歧 >= 50% 時強制 Hold，顯示分歧程度 (如 2/2) / Forces Hold when disagreement >= 50%, shows level (e.g., 2/2) |
| **CatBoost 優化 / CatBoost Optimization** | 降低記憶體使用 (depth 4-6, iterations 200-300, early_stopping_rounds=30, thread_count=4) / Reduced memory usage |
| **GPU 支援 / GPU Support** | 新增 `USE_GPU` 環境變數，可選擇使用 GPU 加速 CatBoost 訓練 / Added `USE_GPU` env var for optional GPU acceleration |
| **每日批次檔 / Daily Batch** | 改進 `run_daily.bat` 顯示進度訊息，修復 Windows 相容性問題 / Improved progress display, fixed Windows compatibility |
| **單元測試 / Unit Tests** | 新增 51 個測試，覆蓋設定、特徵工程、模型訓練 (含 early stopping)、預測功能 / 51 tests covering config, feature engineering, training (incl. early stopping), prediction |
| **新增環境變數 / New Env Vars** | `USE_CATBOOST=True`, `USE_BLENDING=False`, `USE_GPU=False` |

### 修改的檔案 / Modified Files
- `src/train_model.py` — CatBoost、Blending、平行訓練、模型比較表、GPU 偵測、記憶體優化 / CatBoost, Blending, parallel training, model comparison, GPU detection, memory optimization
- `src/predict_upload.py` — 平行預測多支股票、模型分歧計算、特徵對齊修正 / Parallel prediction, model disagreement calculation, feature alignment fix
- `config.py` — 新增 USE_CATBOOST、USE_BLENDING、USE_GPU 環境變數 / Added USE_CATBOOST, USE_BLENDING, USE_GPU env vars
- `run_daily.bat` — 改進進度顯示、修復 Windows 相容性 / Improved progress display, fixed Windows compatibility
- `.env.example` — 新增 USE_GPU 文件 / Added USE_GPU documentation
- `requirements.txt` — 新增 catboost>=1.2.0、pytest>=8.0.0
- `app/streamlit_app.py` — 顯示模型分歧警告 / Displays model disagreement warning
- `migrate_disagreement.sql` — 新增資料庫遷移腳本 / New DB migration script
- `tests/` — 新增測試目錄與 4 個測試檔案 / New test directory and 4 test files

### 改進項目 (2026-09-08) / Improvements (2026-09-08)

| 改進 / Improvement | 說明 / Description |
|---|---|
| **真實準確度驗證 / True Accuracy** | model_monitoring.py 現在使用實際價格數據驗證預測，而非僅用信心度作為代理 / model_monitoring.py now uses actual price data for validation, not confidence as proxy |
| **回測引擎重寫 / Backtest Rewrite** | Backtester 現在實際模擬策略，計算真實回報、勝率、Sharpe Ratio / Backtester now actually simulates strategy, calculates real returns, win rate, Sharpe |
| **Win Rate 修正 / Win Rate Fix** | 模擬結束時若仍持有股票，會計算未實現盈虧並計入勝率 / Unrealized P/L counted toward win rate when holding at simulation end |
| **風險指標 / Risk Metrics** | 新增 Sharpe Ratio、Sortino Ratio、利潤因子、平均持倉天數 / Added Sharpe, Sortino, Profit Factor, Avg Holding Days |
| **基準對比 / Benchmark** | 新增買入持有 (Buy & Hold) 基準，顯示策略超額報酬 (Alpha) / Added Buy & Hold baseline, shows excess return (Alpha) |
| **預測準確度儀表板 / Accuracy Dashboard** | 顯示各股票各時間範圍的真實預測準確度，含長條圖 / True accuracy per stock and timeframe with bar chart |
| **組合模擬 / Portfolio Sim** | 多股票組合模擬，資金平均分配，顯示組合表現 / Multi-stock portfolio, equal allocation, shows performance |
| **信心度加權策略 / Confidence Weighting** | 根據信號信心度調整倉位大小 (30%-100%)，高信心=大倉位 / Position sizing by confidence (30%-100%), high confidence = larger |
| **蒙地卡羅測試 / Monte Carlo** | 隨機翻轉信號 1000 次，顯示報酬分佈、獲利機率、風險指標 / Random signal flips ×1000, shows distribution, probability, risk metrics |
| **yfinance 優先 / yfinance Priority** | 資料來源改為 yfinance 優先，akshare 作為備援 / yfinance as primary data source, akshare as fallback |

### 修改的檔案 / Modified Files
- `src/simulator.py` — 新增風險指標、基準對比、組合模擬、信心度加權、蒙地卡羅 / Added risk metrics, benchmark, portfolio, confidence weighting, Monte Carlo
- `src/model_monitoring.py` — 重寫 calculate_accuracy 使用真實價格驗證、重寫回測引擎 / Rewrote calculate_accuracy with real price verification, rewrote backtest engine
- `app/pages/1_💰_投資模擬器.py` — 新增風險指標顯示、基準對比、準確度分析、進階功能 / Added risk metrics display, benchmark comparison, accuracy analysis, advanced features
- `src/data_fetcher.py` — yfinance 優先，akshare 備援 / yfinance priority, akshare fallback

### 改進項目 (2026-09-04) / Improvements (2026-09-04)

| 改進 / Improvement | 說明 / Description |
|---|---|
| **投資模擬器 / Investment Simulator** | 新增 Streamlit 模擬頁面，可自訂日期範圍、資金、時間範圍，模擬跟單收益 / New Streamlit simulator page, custom date range, capital, timeframe |
| **交易成本 / Trading Costs** | 模擬包含佣金 0.1% + 印花稅 0.13%，計算淨收益 / Simulation includes 0.1% commission + 0.13% stamp duty |
| **多頁面導航 / Multi-page Nav** | Streamlit 側邊欄自動顯示頁面導航 (預測儀表板 + 投資模擬器) / Sidebar auto-shows page nav (Dashboard + Simulator) |

### 修改的檔案 / Modified Files
- `src/simulator.py` — 新增投資模擬引擎 / New investment simulation engine
- `app/pages/1_💰_投資模擬器.py` — 新增投資模擬互動頁面 / New interactive simulator page

### 改進項目 (2026-09-03) / Improvements (2026-09-03)

| 改進 / Improvement | 說明 / Description |
|---|---|
| **日誌系統 / Logging** | 全面使用 `setup_logger()` 取代 `print()`，統一日誌格式 / Replaced all `print()` with `setup_logger()`, unified log format |
| **單例模式 / Singleton** | Supabase 客戶端改為單例模式，避免重複連接 / Supabase client now singleton, avoids duplicate connections |
| **勝率計算修正 / Win Rate Fix** | `get_win_rate()` 改為驗證實際價格變動 (之前返回虛假數據) / `get_win_rate()` now validates actual price changes (was returning fake data) |
| **MFI 向量化 / MFI Vectorization** | MFI 計算從 Python 迴圈改為 NumPy 向量化運算 (速度提升 ~10x) / MFI from Python loop to NumPy vectorized (~10x speedup) |
| **環境變數檢查 / Env Check** | 啟動時自動檢查 `.env` 檔案是否存在，缺失時提供清楚錯誤訊息 / Auto-checks `.env` on startup, clear error message if missing |
| **依賴版本鎖定 / Dependency Pinning** | requirements.txt 加入版本上限 (`<2.0.0`)，防止未來不相容更新 / Added version caps (`<2.0.0`) to prevent future incompatibility |
| **型別標註 / Type Hints** | cleanup_old.py, predict_upload.py 新增型別提示，提升程式碼可維護性 / Added type hints for better maintainability |

### 修改的檔案 / Modified Files
- `config.py` — 新增 `.env` 檔案存在檢查 / Added `.env` file existence check
- `src/cleanup_old.py` — 使用 logger、型別標註、新增刪除前記錄筆數顯示 / Logger, type hints, pre-delete count display
- `src/predict_upload.py` — 單例 Supabase 客戶端、修正勝率計算、型別標註 / Singleton Supabase client, fixed win rate calc, type hints
- `src/feature_engineering.py` — MFI 計算向量化 / MFI vectorized computation
- `src/train_model.py` — 新增文件字串 / Added docstrings
- `requirements.txt` — 版本鎖定、分類整理 / Version pinning, organized by category

## 授權 / License

本專案僅供學習和研究使用，不構成任何投資建議。投資有風險，入市需謹慎。 / This project is for educational and research purposes only. It does not constitute any investment advice. Investing involves risk — invest with caution.

> **Warning:** 過去表現不代表未來結果。本系統的所有回測和模擬結果僅供參考，不應作為實際投資決策的依據。 / Past performance does NOT guarantee future results. All backtest and simulation results from this system are for reference only and should NOT be used as the basis for actual investment decisions.
