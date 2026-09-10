# 港股每日自動預測系統

純機器學習的港股每日預測系統，使用 XGBoost / LightGBM / RandomForest / CatBoost 集成模型進行漲跌預測，結果自動上傳至 Supabase 雲端資料庫，並透過 Streamlit 網站展示。

## 系統架構

```
Windows 本地定時訓練 (平行) → 預測結果上傳至 Supabase (PostgreSQL) → Streamlit 網站顯示
```

## 功能特色

### 核心功能
- **多時間範圍預測**: 同時預測明日(1天)、下週(5天)、下月(20天)
- **四模型集成**: XGBoost + LightGBM + RandomForest + CatBoost
- **三種集成模式**: Voting (加權平均) / Stacking (元模型) / Blending (out-of-fold)
- **平行訓練**: 時間範圍同時訓練，速度提升 ~3x
- **平行預測**: 多支股票同時預測
- **SMOTE 類別平衡**: 自動處理正負樣本不平衡問題
- **33項技術指標**: 新增動量、波動率、威廉指標、MFI 等
- **特徵相關性過濾**: 自動移除 |corr| > 0.9 的冗餘特徵
- **閾值優化**: 自動搜尋最佳 Buy/Sell 信心度閾值 (取代固定 0.55/0.45)
- **模型版本化**: 帶時間戳的模型備份，自動保留最近 5 版，支持回滾
- **模型指標追蹤**: 記錄 F1 Score、AUC Score、冠軍模型類型
- **模型分歧檢測**: 當模型意見分歧 >= 50% 時強制 Hold，顯示分歧程度
- **GPU 支援**: CatBoost 可選擇使用 GPU 加速 (透過 `USE_GPU=True` 啟用)
- **互動式儀表板**: Streamlit 顯示預測結果、信心度趨勢、信號分佈
- **K線圖 (含買賣信號)**: 互動式K線圖，標示模型預測的買入/賣出信號位置，支援自選股票和時間範圍
- **技術指標展示**: 顯示 RSI、MACD、Stochastic、ADX、MFI、布林帶寬、ATR、量比等最新數值，作為信號依據
- **閾值互動控制**: 圖表可選擇時間範圍顯示 Buy/Sell 閾值線，避免多線重疊
- **數據匯出**: 支援 CSV 和 Excel 格式匯出預測記錄
- **投資模擬器**: 根據 Buy/Sell 信號模擬投資，計算實際收益、勝率、最大回撤，支援自訂資金與日期範圍
- **風險指標**: Sharpe Ratio、Sortino Ratio、利潤因子、平均持倉天數
- **基準對比**: 策略報酬 vs 買入持有 (Buy & Hold) 對比，顯示超額報酬 (Alpha)
- **預測準確度分析**: 驗證歷史預測是否正確，顯示各股票各時間範圍的真實準確度
- **組合模擬**: 多股票組合模擬，顯示組合表現和 diversification 效果
- **信心度加權策略**: 根據信號信心度調整倉位大小 (高信心=大倉位)
- **蒙地卡羅測試**: 隨機翻轉信號 1000 次，測試策略穩健性，顯示報酬分佈

### 風險管理
- **止損/止盈建議**: 基於波動率自動計算建議止損止盈點
- **風險報酬比**: 評估潛在收益與風險的比例
- **預期報酬**: 基於信心度和波動率估算預期報酬率
- **信心度追蹤**: 顯示信心度變化趨勢 (↑↓→)
- **勝率統計**: 歷史預測準確率追蹤
- **模型分歧檢測**: 當模型意見分歧時自動 Hold，避免弱勢決策

### 模型監控
- **數據品質檢查**: 自動檢測缺失日期、信心度分佈異常
- **真實準確度驗證**: 透過實際價格數據驗證預測是否正確 (非僅信心度代理)
- **滾動準確度追蹤**: 30天滾動窗口計算真實預測準確度 (Buy後價格是否上漲？Sell後價格是否下跌？)
- **訓練指標展示**: 顯示各股票各時間範圍的 F1 Score、AUC Score、冠軍模型類型
- **模型漂移檢測**: 監控模型性能是否下降 (>10% = 高度, >5% = 中度)
- **回測引擎**: 實際模擬策略表現，計算真實回報、勝率、Sharpe Ratio
- **信號警報**: 強勢信號自動提醒 (信心度>70% 或 預期報酬>5%)
- **信心度校準**: 確保信心度分數可靠

## 快速開始

### 1. 安裝依賴

```bash
# 方法一：使用批次檔一鍵安裝
setup.bat

# 方法二：手動安裝
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 2. 配置環境變數

1. 複製 `.env.example` 為 `.env`
2. 填入你的 Supabase 專案資訊：

```env
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-anon-key
STOCK_LIST=0700,9988,0005,0939

# 模型訓練開關
USE_ENSEMBLE=True
USE_STACKING=False
USE_BLENDING=False
USE_CATBOOST=True
USE_SMOTE=True
USE_GPU=False
```

**⚠️ 重要：** 先在 [Supabase 官網](https://supabase.com) 取得專案 URL 與金鑰，填入 `.env` 後再執行。

### 3. 初始化資料庫

```bash
python src/init_database.py
```

### 4. 訓練模型

```bash
python src/train_model.py
```

訓練完成後會顯示：
- 各時間範圍的冠軍模型 (xgboost/lightgbm/catboost/voting/stacking/blending)
- 模型比較表 (各模型 F1 分數)
- F1 Score 和 AUC Score
- Top 10 特徵重要性

### 5. 每日預測與上傳

```bash
python src/predict_upload.py
```

### 6. 啟動預測儀表板

```bash
streamlit run app/streamlit_app.py
```

儀表板包含兩個頁面 (側邊欄切換)：
- **📈 預測儀表板**: 信號卡片、技術指標、K線圖 (含買賣信號)、信心度趨勢、信號分佈、預測記錄、滾動準確度、訓練指標
- **💰 投資模擬器**: 自訂日期範圍、資金、時間範圍，模擬跟單收益

### 7. 執行測試

```bash
# 執行所有測試
python -m pytest tests/ -v

# 執行特定測試檔案
python -m pytest tests/test_config.py -v

# 執行特定測試類別
python -m pytest tests/test_feature_engineering.py::TestFeatureEngineering -v

# 執行特定測試函數
python -m pytest tests/test_train_model.py::TestModelTraining::test_train_xgboost -v

# 顯示詳細資訊
python -m pytest tests/ -v --tb=long

# 只顯示失敗的測試
python -m pytest tests/ -v --tb=short
```

**測試覆蓋範圍：**
| 測試檔案 | 測試數量 | 覆蓋範圍 |
|---|---|---|
| `test_config.py` | 10 | 環境變數、股票列表解析、Supabase 設定 |
| `test_feature_engineering.py` | 17 | RSI、MACD、Bollinger、ATR、ADX、Stochastic、MFI、Williams %R |
| `test_train_model.py` | 14 | XGBoost、LightGBM、RandomForest、CatBoost (含 early stopping)、SMOTE、Blending |
| `test_predict.py` | 10 | 預測日期、模型載入、信號判定、上傳功能 |
| **總計** | **51** | |

## 設定 Windows 自動排程

使用 Windows 工作排程器，設定每日 16:30（港股收盤後）自動執行：

1. 按 `Win + R`，輸入 `taskschd.msc` 開啟工作排程器
2. 點擊右側「建立基本工作...」
3. 名稱：`港股每日預測`
4. 觸發器：選擇「每日」，開始時間設為 `16:30`
5. 動作：選擇「啟動程式」
6. 程式或指令：瀏覽選擇 `run_daily.bat`
7. 完成後，右鍵該工作 → 內容 → 設定：
   - ✅ 喚醒電腦執行此工作
   - ✅ 不論使用者是否登入都要執行

## 專案結構

```
project_root/
├── .env.example          # 環境變數範例
├── .gitignore            # Git 忽略清單
├── requirements.txt      # Python 依賴
├── config.py             # 讀取 .env，提供全域設定
├── run_daily.bat         # Windows 批次檔 (排程器用)
├── setup.bat             # 一鍵安裝依賴
├── logs/                 # 日誌資料夾
├── models/               # 訓練好的模型 (.pkl)
│   ├── best_model_{tf}.pkl           # 當前模型
│   ├── best_model_{tf}_{ts}.pkl      # 版本化模型 (保留最近5版)
│   ├── feature_importance_{tf}.csv   # 特徵重要性
│   └── roc_curve_{tf}.png            # ROC 曲線
├── tests/                # 單元測試
│   ├── __init__.py
│   ├── test_config.py           # 設定模組測試
│   ├── test_feature_engineering.py  # 特徵工程測試
│   ├── test_train_model.py      # 模型訓練測試
│   └── test_predict.py          # 預測上傳測試
├── src/
│   ├── __init__.py
│   ├── logger.py         # 日誌設定
│   ├── init_database.py  # 自動建表 (冪等)
│   ├── data_fetcher.py   # 下載港股歷史數據 (akshare/yfinance)
│   ├── feature_engineering.py  # 33項技術指標計算
│   ├── train_model.py    # Optuna 自動調參 + Voting/Stacking 集成 + SMOTE
│   ├── predict_upload.py # 每日預測並上傳 Supabase
│   ├── simulator.py      # 投資模擬引擎 (信號模擬、交易成本、績效追蹤)
│   ├── cleanup_old.py    # 清理舊數據 (保留60天)
│   └── model_monitoring.py  # 數據品質、模型漂移、警報、校準
├── app/
│   ├── __init__.py
│   ├── streamlit_app.py  # Streamlit 預測儀表板
│   └── pages/
│       └── 1_💰_投資模擬器.py  # 投資模擬互動頁面
├── migrate_metrics.sql   # 資料庫遷移: 模型指標欄位
├── migrate_quick_wins.sql # 資料庫遷移: 風險管理欄位
├── migrate_thresholds.sql # 資料庫遷移: Buy/Sell 閾值欄位
└── migrate_disagreement.sql # 資料庫遷移: 模型分歧指標
```

## 技術細節

### 機器學習模型
- **演算法**: XGBoost + LightGBM + RandomForest + CatBoost 集成
- **集成方式**: VotingClassifier (soft voting) 或 StackingClassifier (元模型 = LogisticRegression) 或 Blending (out-of-fold)
- **超參數優化**: Optuna (50 trials，同時搜尋四個模型 + voting 權重)
- **權重優化**: Optuna 自動搜尋最佳權重組合 (如 [0.3, 0.3, 0.2, 0.2])，非固定 1:1:1:1
- **交叉驗證**: TimeSeriesSplit (n_splits=5)，嚴格遵守時序，不洩漏未來資訊
- **類別不平衡處理**: SMOTE (僅在訓練折上套用，不跨越驗證折)
- **訓練數據**: 3 年歷史數據 (約 750 交易日)
- **評估指標**: F1 Score, AUC, Precision, Recall
- **ROC 曲線**: 自動儲存至 `models/roc_curve_{timeframe}.png`
- **特徵相關性過濾**: 自動移除 |corr| > 0.9 的冗餘特徵
- **閾值優化**: 自動搜尋最佳 Buy/Sell 信心度閾值 (取代固定 0.55/0.45)
- **模型版本化**: 帶時間戳備份，自動保留最近 5 版
- **特徵重要性**: 輸出至 `models/feature_importance_{timeframe}.csv`
- **模型分歧檢測**: 當四個模型意見分歧 >= 50% 時強制 Hold
- **特徵對齊**: 訓練時保存 `feature_columns`，預測時嚴格使用相同順序
- **CatBoost 早停**: CatBoost 使用 early_stopping_rounds=30，搭配 eval_set 驗證集，自動停止訓練避免過擬合 (iterations 200-300)
- **GPU 支援**: CatBoost 可選擇使用 GPU 加速 (透過 `USE_GPU=True` 啟用)

### 技術指標 (33 Features)

| 類別 | 特徵 | 說明 |
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

### 模型訓練開關

| 環境變數 | 預設值 | 說明 |
|---|---|---|
| `USE_ENSEMBLE` | `True` | 啟用模型集成 (False = 單模型比較) |
| `USE_STACKING` | `False` | 使用 StackingClassifier (元模型學習組合) |
| `USE_BLENDING` | `False` | 使用 Blending (out-of-fold stacking，通常更準確) |
| `USE_CATBOOST` | `True` | 包含 CatBoost 作為第4個模型 |
| `USE_SMOTE` | `True` | 啟用 SMOTE 類別不平衡處理 |
| `USE_GPU` | `False` | CatBoost 使用 GPU 訓練 (需要 NVIDIA GPU，會使用更多記憶體) |

**優先級規則：**
- `USE_STACKING=True` 或 `USE_BLENDING=True` → 強制使用集成模式
- `USE_ENSEMBLE=True` (預設) → VotingClassifier (加權平均)
- `USE_ENSEMBLE=False` → 單一最佳模型 (XGBoost vs LightGBM vs CatBoost)

**訓練速度：**
- 時間範圍 (1d, 5d, 20d) **平行訓練**，速度提升 ~3x
- 多支股票預測也支援**平行處理**

### 目標變數 (Target)
- **目標**: N天後收盤價 > 今日收盤價 → 1 (Buy)，否則 → 0
- **類別權重**: 自動平衡正負樣本 (上限3倍) + SMOTE 擴充
- **多時間範圍**: 1天、5天、20天

### 模型表現 (F1 Score)
| 時間範圍 | F1 Score | 說明 |
|---|---|---|
| 1天 | ~0.57 | 可用 — 短期趨勢 |
| 5天 | ~0.69 | 良好 — 中期動量 |
| 20天 | ~0.73 | 最佳 — 長期趨勢 |

**注意**: 股票預測本身非常困難，AUC ~0.55-0.60 已是合理範圍。

### 信號判定
- 閾值由模型自動優化 (在驗證集上搜尋最佳 F1)，以下為預設值：
| 信心度 | 信號 |
|---|---|
| > 閾值 (優化後，預設55%) | Buy (買入) |
| < 閾值 (優化後，預設45%) | Sell (賣出) |
| 其餘 | Hold (持有) |

### 風險管理指標

| 指標 | 說明 | 計算方式 |
|---|---|---|
| **預期報酬** | 基於信心度和波動率估算 | `(信心度-0.5) × 2 × 波動率 × √天數` |
| **止損點** | 建議止損位置 | `2 × 波動率 × √天數` |
| **止盈點** | 建議止盈位置 | `1.5 × \|預期報酬\|` |
| **風險報酬比** | 收益與風險比例 | `報酬 / 風險` |
| **信心度趨勢** | 信心度變化方向 | ↑上升 ↓下降 →持平 |
| **勝率** | 歷史預測準確率 | `Buy+Sell信號比例` |

### 資料來源
- **股票數據**: yfinance (主) / akshare (備)
- **市場指數**: yfinance (^HSI 恒生指數)
- **匯率**: yfinance (USD/HKD)

### 投資模擬

根據歷史預測信號模擬跟單交易，計算實際投資收益。

| 項目 | 說明 |
|---|---|
| **初始資金** | 每檔股票 HKD 20,000 (可在頁面自訂) |
| **Buy 信號** | 以當日收盤價買入最大可購入股數 (整股) |
| **Sell 信號** | 以當日收盤價賣出所有持股 |
| **Hold 信號** | 不進行任何交易 (模擬結束時 counted as win/loss) |
| **不做空** | Sell 信號僅用於平倉，不做空 |

**交易成本 (香港標準)：**

| 費用 | 比例 | 說明 |
|---|---|---|
| 佣金 | 0.1% | 每筆交易最低 HKD 20 |
| 印花稅 | 0.13% | 僅賣出時收取 |

**計算公式：**
```
買入股數 = floor(可投資金額 / (股價 × (1 + 佣金率)))
賣出所得 = 股數 × 股價 × (1 - 佣金率 - 印花稅率)
盈虧 = 賣出所得 - 買入成本
```

**輸出指標：**

| 指標 | 說明 |
|---|---|
| 總盈虧 (HKD) | 投資組合最終價值 - 初始資金 |
| 報酬率 (%) | (最終價值 - 初始資金) / 初始資金 × 100 |
| 交易次數 | 買入 + 賣出次數 |
| 勝率 (%) | 盈利交易次數 / 總交易次數 × 100% (含持倉到期) |
| 最大回撤 (%) | 歷史最高點到最低點的跌幅百分比 |
| Sharpe Ratio | 年化風險調整報酬率 (>1 不錯, >2 很好) |
| Sortino Ratio | 只考慮下跌風險的風險調整報酬率 |
| 利潤因子 | 總盈利 / 總虧損 (>1 表示盈利大於虧損) |
| 平均持倉天數 | 每次買入到賣出的平均天數 |
| 買入持有報酬 | 基準策略：在開始時買入並持有到結束 |
| 超額報酬 (Alpha) | 策略報酬 - 買入持有報酬 |

**進階功能：**

| 功能 | 說明 |
|---|---|
| **組合模擬** | 模擬所有選中股票的組合表現，資金平均分配，顯示組合 Sharpe 和最大回撤 |
| **信心度加權** | 根據信號信心度調整倉位大小 (30%-100% 資金)，高信心=大倉位 |
| **蒙地卡羅測試** | 隨機翻轉信號 1000 次，測試策略穩健性，顯示報酬分佈和獲利機率 |
| **預測準確度** | 驗證歷史預測是否正確：Buy 後價格是否上漲？Sell 後價格是否下跌？ |
| **最佳時機模式** | 假設預知未來 N 天價格，選最佳買賣日 (後見之明模式) |

## 資料庫結構

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
    f1_score FLOAT8,          -- 模型 F1 分數
    auc_score FLOAT8,         -- 模型 AUC 分數
    expected_return FLOAT8,   -- 預期報酬率 (%)
    risk_reward FLOAT8,       -- 風險報酬比
    stop_loss FLOAT8,         -- 止損點 (%)
    take_profit FLOAT8,       -- 止盈點 (%)
    confidence_trend TEXT,    -- 信心度趨勢: ↑↓→-
    win_rate FLOAT8,          -- 歷史勝率 (%)
    threshold_buy FLOAT8,     -- 優化後的 Buy 閾值 (每個時間範圍不同)
    threshold_sell FLOAT8,    -- 優化後的 Sell 閾值 (每個時間範圍不同)
    model_disagreement FLOAT8, -- 模型分歧度 (0=一致, 0.5=2v2, 1=完全分歧)
    model_split TEXT,          -- 模型投票結果 (e.g., '4/0', '3/1', '2/2')
    created_at TIMESTAMPTZ DEFAULT NOW()
);
```

### 資料庫遷移

執行以下 SQL 語句來添加新欄位：

```sql
-- 模型指標欄位
ALTER TABLE stock_predictions ADD COLUMN IF NOT EXISTS model_type TEXT;
ALTER TABLE stock_predictions ADD COLUMN IF NOT EXISTS f1_score FLOAT8;
ALTER TABLE stock_predictions ADD COLUMN IF NOT EXISTS auc_score FLOAT8;

-- 風險管理欄位
ALTER TABLE stock_predictions ADD COLUMN IF NOT EXISTS expected_return FLOAT8;
ALTER TABLE stock_predictions ADD COLUMN IF NOT EXISTS risk_reward FLOAT8;
ALTER TABLE stock_predictions ADD COLUMN IF NOT EXISTS stop_loss FLOAT8;
ALTER TABLE stock_predictions ADD COLUMN IF NOT EXISTS take_profit FLOAT8;
ALTER TABLE stock_predictions ADD COLUMN IF NOT EXISTS confidence_trend TEXT DEFAULT '-';
ALTER TABLE stock_predictions ADD COLUMN IF NOT EXISTS win_rate FLOAT8;

-- 模型分歧指標
ALTER TABLE stock_predictions ADD COLUMN IF NOT EXISTS model_disagreement FLOAT8 DEFAULT 0;
ALTER TABLE stock_predictions ADD COLUMN IF NOT EXISTS model_split TEXT DEFAULT '0/0';

-- 移除唯一限制 (保留歷史記錄)
ALTER TABLE stock_predictions DROP CONSTRAINT IF EXISTS unique_stock_prediction;
```

## 模型監控

### 數據品質檢查
- 檢測缺失日期 (每個時間範圍至少需要 5 筆預測)
- 檢查信心度分佈是否合理
- 檢測信號分佈是否異常 (某信號 >70%)

### 真實準確度驗證
- 透過 yfinance 獲取實際價格數據
- Buy 信號：N天後收盤價 > 預測日收盤價 = 正確
- Sell 信號：N天後收盤價 < 預測日收盤價 = 正確
- 顯示各股票、各時間範圍的真實準確度百分比

### 模型漂移檢測
- 比較最近 7 天 vs 30 天的預測準確度 (使用真實價格驗證)
- 準確度下降 >10% = 高度警報
- 準確度下降 >5% = 中度警報

### 回測引擎
- 實際模擬策略表現：買入 → 賣出 → 計算真實回報
- 計算 Sharpe Ratio、勝率、最大回撤
- 對比買入持有基準，計算超額報酬 (Alpha)

### 信號警報
- 信心度 >70% 的強勢信號
- 預期報酬 >5% 的高回報信號

### 信心度校準
- 監控平均信心度是否合理
- 過度自信 (>60%) 或信心不足 (<40%) 會建議調整

## 注意事項

- 所有日期時間使用香港時區 (`Asia/Hong_Kong`)
- 特徵計算嚴禁 Look-ahead Bias（只使用當天之前的數據）
- `.env` 檔案包含敏感資訊，請勿上傳至版本控制
- 模型檔案 (`.pkl`) 不上傳至版本控制
- 每日預測會保留歷史記錄 (自動清理 60 天前的舊數據)

## 常見問題

### Q: 為什麼 AUC 只有 0.55-0.60？
A: 股票預測本身非常困難。即使是大型對沖基金，AUC 也通常在 0.55-0.65 之間。你的模型已達到合理範圍。

### Q: F1 Score 代表什麼？
A: F1 = 精準率與召回率的平衡。F1 > 0.5 表示模型比隨機好，F1 > 0.6 表示可用於交易信號。

### Q: 什麼是模型集成 (Ensemble)？
A: 同時訓練 XGBoost、LightGBM、RandomForest、CatBoost 四個模型，透過 VotingClassifier (加權平均)、StackingClassifier (元模型學習) 或 Blending (out-of-fold stacking) 結合它們的預測機率。通常比單一模型更穩定、AUC 更高。

### Q: Voting、Stacking、Blending 有什麼差別？
A: 
- **Voting**: 用加權平均結合四個模型的預測機率 (預設，最快)
- **Stacking**: 用一個元模型 (LogisticRegression) 學習如何最佳組合四個模型的預測 (較慢但通常更準)
- **Blending**: 類似 Stacking，但使用 out-of-fold predictions 避免過擬合 (最慢但通常最準)

### Q: CatBoost 是什麼？為什麼要加它？
A: CatBoost 是 Yandex 開發的梯度提升框架，對類別型特徵處理更好，在金融數據上通常表現優於 XGBoost/LightGBM。加入後可提升集成模型的準確度。

### Q: SMOTE 是什麼？為什麼需要它？
A: SMOTE (Synthetic Minority Over-sampling Technique) 在訓練集上生成少數類的合成樣本，解決正負樣本不平衡問題。僅在 TimeSeriesSplit 的訓練折上套用，不會洩漏未來資訊。

### Q: 可以添加更多股票嗎？
A: 修改 `.env` 中的 `STOCK_LIST`，例如 `STOCK_LIST=0700,9988,0005,0939,1810`

### Q: 如何查看訓練日誌？
A: 日誌位於 `logs/app.log`

### Q: 訓練要多久？
A: 約 3-5 分鐘 (使用平行訓練，時間範圍同時訓練)。若使用 Stacking 或 Blending，約 5-8 分鐘。

### Q: 預測要多久？
A: 平行預測多支股票，約 5-10 秒 (取決於股票數量)。

### Q: 什麼是特徵相關性過濾？
A: 訓練前自動移除 |corr| > 0.9 的冗餘特徵。例如 `ret_3d` 與 `ret_1d`/`ret_5d` 高度相關，只保留最具資訊量的一個。減少噪音、加快訓練、降低過擬合。

### Q: 閾值優化是什麼？
A: 一般系統用固定閾值 (Buy > 55%, Sell < 45%)，但不同時間範圍的最佳閾值不同。系統會在驗證集上自動搜尋使 F1 最高的 Buy/Sell 閾值，訓練後存入模型。

### Q: 模型版本化有什麼用？
A: 每次訓練會保存帶時間戳的模型備份 (如 `best_model_5d_20260822_163000.pkl`)，自動保留最近 5 版。如果新模型效果不好，可以手動回滾到舊版。

### Q: 特徵重要性 CSV 怎麼用？
A: 訓練後自動輸出至 `models/feature_importance_{timeframe}.csv`。可用 Excel 開啟分析哪些特徵對模型預測最有影響，協助特徵工程優化。

### Q: 模型會自動更新嗎？
A: 需要手動執行 `train_model.py` 重新訓練，或設定 Windows 排程器自動執行

### Q: 如何解讀止損/止盈點？
A: 
- Buy 信號：止損為負數 (下跌止損)，止盈為正數 (上漲獲利)
- Sell 信號：止損為正數 (上漲止損)，止盈為負數 (下跌獲利)

### Q: 模型漂移是什麼？
A: 模型漂移是指模型預測能力隨時間下降。系統會自動檢測並提醒您重新訓練。

### Q: 信心度校準有什麼用？
A: 確保信心度分數可靠。如果模型過度自信或信心不足，系統會建議調整。

### Q: 投資模擬器是什麼？
A: 投資模擬器根據歷史預測信號 (Buy/Sell/Hold) 模擬跟單交易。每檔股票以 HKD 20,000 初始資金，Buy 時買入、Sell 時賣出，計算實際收益、勝率和最大回撤。可在 Streamlit 側邊欄切換至「💰 投資模擬器」頁面使用。

### Q: 模擬結果包含交易成本嗎？
A: 是的。模擬包含香港標準交易成本：佣金 0.1% (每筆最低 HKD 20) + 印花稅 0.13% (僅賣出)。計算出的盈虧為扣除成本後的淨收益。

### Q: 模擬器可以做空嗎？
A: 不可以。Sell 信號僅用於賣出已持有的股票 (平倉)，不進行做空操作。若沒有持股時出現 Sell 信號，則不執行任何交易。

### Q: 什麼是 Sharpe Ratio？
A: Sharpe Ratio 是年化風險調整報酬率。計算方式為 (平均報酬 - 無風險利率) / 報酬標準差 × √252。一般來說：>1 表示不錯，>2 表示很好，<0 表示不如無風險投資。

### Q: 什麼是 Sortino Ratio？
A: Sortino Ratio 與 Sharpe Ratio 類似，但只考慮下跌風險 (負報酬的標準差)。適合評估不對稱報酬的投資策略。

### Q: 什麼是利潤因子？
A: 利潤因子 = 總盈利 / 總虧損。>1 表示盈利大於虧損，>2 表示很好。如果利潤因子 < 1，表示策略整體虧損。

### Q: 什麼是買入持有基準？
A: 買入持有 (Buy & Hold) 是最簡單的投資策略：在開始時買入並持有到結束。系統會將信號策略的報酬與買入持有對比，顯示超額報酬 (Alpha)。正 Alpha 表示策略跑贏大盤。

### Q: 什麼是組合模擬？
A: 組合模擬將資金平均分配到所有選中的股票，模擬多股票組合的表現。這可以顯示分散投資的效果，以及組合的整體 Sharpe Ratio 和最大回撤。

### Q: 什麼是信心度加權策略？
A: 信心度加權策略根據信號的信心度調整倉位大小。高信心信號 (如 90%) 會投入更多資金 (最多 100%)，低信心信號 (如 50%) 只投入較少資金 (最少 30%)。這可以讓你在更有把握的交易上投入更多。

### Q: 什麼是蒙地卡羅測試？
A: 蒙地卡羅測試隨機翻轉信號 (如 20% 的 Buy/Sell 信號被隨機交換)，然後運行 1000 次模擬。這可以測試策略的穩健性：如果策略在信號被隨機干擾後仍然盈利，表示策略較為可靠。

### Q: 預測準確度如何計算？
A: 系統會驗證歷史預測是否正確：Buy 信號 → N天後收盤價是否上漲？Sell 信號 → N天後收盤價是否下跌？準確度 = 正確預測數 / 總預測數 × 100%。這使用真實價格數據驗證，而非僅用信心度作為代理。

### Q: 勝率和預測準確度有什麼差別？
A: 勝率是指模擬交易中盈利的交易比例 (賣出或持倉到期時計算)。預測準確度是指信號方向是否正確 (價格是否朝預測方向移動)。兩者可能不同，因為勝率還受到交易成本、進出場時機等因素影響。

### Q: 什麼是模型分歧 (Model Disagreement)？
A: 模型分歧是指四個模型 (XGBoost, LightGBM, RandomForest, CatBoost) 預測結果不一致的程度。例如：
- `4/0`：四個模型都認為 Buy → 分歧度 0 (完全一致)
- `3/1`：三個 Buy，一個 Sell → 分歧度 0.25
- `2/2`：兩個 Buy，兩個 Sell → 分歧度 0.5 (五五波)
系統會在分歧度 >= 50% 時強制將信號設為 Hold，避免在模型意見分歧時做出弱勢決策。

### Q: 模型分歧時為什麼要強制 Hold？
A: 當模型意見分歧時 (如 2v2)，表示市場方向不明確。此時做出 Buy 或 Sell 決策風險較高。強制 Hold 可以避免在不確定時刻進場，保護資金安全。儀表板會顯示 ⚠️ 警告標示分歧程度。

### Q: K線圖上的買賣信號怎麼看？
A: K線圖上標示了模型預測的買入 (▲ 綠色) 和賣出 (▼ 紅色) 信號位置。三角形位置對應預測日的價格，您可以直觀看到：信號發出後價格是否朝預測方向移動。例如 Buy 信號後價格上漲，表示預測正確。

### Q: 技術指標的作用是什麼？
A: 技術指標 (RSI、MACD、Stochastic 等) 是模型學習的輸入特徵。儀表板顯示這些指標的最新數值，讓您了解：模型做出 Buy/Sell 決策時，技術面是否支持這個判斷。例如 RSI > 70 (超買) 時出現 Buy 信號，可能需要謹慎。

### Q: 滾動準確度怎麼計算？
A: 滾動準確度 = 最近 30 天內正確預測數 / 總預測數 × 100%。驗證方式：Buy 信號 → N天後收盤價是否上漲？Sell 信號 → N天後收盤價是否下跌？這使用真實價格數據，比單看訓練指標 (F1/AUC) 更可靠。

### Q: 如何執行測試？
A: 使用 pytest 執行測試：`python -m pytest tests/ -v`。測試覆蓋環境變數設定、特徵工程、模型訓練、預測上傳等核心功能。

### Q: 測試覆蓋了哪些功能？
A: 共 51 個測試，涵蓋：
- 環境變數載入與驗證 (10 個)
- 技術指標計算：RSI、MACD、ATR、ADX、Stochastic、MFI、Williams %R (17 個)
- 模型訓練：XGBoost、LightGBM、RandomForest、CatBoost (含 early stopping)、SMOTE、Blending (14 個)
- 預測功能：日期計算、模型載入、信號判定、上傳 (10 個)

## 近期更新

### 改進項目 (2026-09-11)

| 改進 | 說明 |
|---|---|
| **K線圖 (含買賣信號)** | 新增互動式K線圖，標示模型預測的買入(▲)/賣出(▼)信號位置，支援自選股票和時間範圍 (30/60/90/180天) |
| **技術指標展示** | 新增技術指標面板，顯示 RSI、MACD、Stochastic、ADX、MFI、布林帶寬、ATR、量比等最新數值，作為信號依據 |
| **滾動準確度追蹤** | 新增30天滾動窗口計算真實預測準確度，驗證 Buy 後價格是否上漲、Sell 後價格是否下跌 |
| **訓練指標展示** | 新增訓練指標面板，顯示各股票各時間範圍的 F1 Score、AUC Score、冠軍模型類型 |

**無資料庫變更** — 所有功能讀取現有 `stock_predictions` 表或即時計算。

### 修改的檔案
- `app/streamlit_app.py` — 新增 K線圖、技術指標、滾動準確度、訓練指標展示

### 改進項目 (2026-09-10)

| 改進 | 說明 |
|---|---|
| **CatBoost 模型** | 新增 CatBoost 作為第4個模型選項，通常在金融數據上表現更好 |
| **Blending 集成** | 新增 Blending 模式 (out-of-fold stacking)，通常比 Voting 更準確 |
| **平行訓練** | 時間範圍 (1d, 5d, 20d) 平行訓練，速度提升 ~3x |
| **平行預測** | 多支股票預測平行處理，大幅提升預測速度 |
| **模型比較表** | 訓練時顯示各模型 F1 分數比較，清楚標示贏家 |
| **模型分歧檢測** | 當模型意見分歧 >= 50% 時強制 Hold，顯示分歧程度 (如 2/2) |
| **CatBoost 優化** | 降低記憶體使用 (depth 4-6, iterations 200-300, early_stopping_rounds=30, thread_count=4) |
| **GPU 支援** | 新增 `USE_GPU` 環境變數，可選擇使用 GPU 加速 CatBoost 訓練 |
| **每日批次檔** | 改進 `run_daily.bat` 顯示進度訊息，修復 Windows 相容性問題 |
| **單元測試** | 新增 51 個測試，覆蓋設定、特徵工程、模型訓練 (含 early stopping)、預測功能 |
| **新增環境變數** | `USE_CATBOOST=True`, `USE_BLENDING=False`, `USE_GPU=False` |

### 修改的檔案
- `src/train_model.py` — CatBoost、Blending、平行訓練、模型比較表、GPU 偵測、記憶體優化
- `src/predict_upload.py` — 平行預測多支股票、模型分歧計算、特徵對齊修正
- `config.py` — 新增 USE_CATBOOST、USE_BLENDING、USE_GPU 環境變數
- `run_daily.bat` — 改進進度顯示、修復 Windows 相容性
- `.env.example` — 新增 USE_GPU 文件
- `requirements.txt` — 新增 catboost>=1.2.0、pytest>=8.0.0
- `app/streamlit_app.py` — 顯示模型分歧警告
- `migrate_disagreement.sql` — 新增資料庫遷移腳本
- `tests/` — 新增測試目錄與 4 個測試檔案

### 改進項目 (2026-09-08)

| 改進 | 說明 |
|---|---|
| **真實準確度驗證** | model_monitoring.py 現在使用實際價格數據驗證預測，而非僅用信心度作為代理 |
| **回測引擎重寫** | Backtester 現在實際模擬策略，計算真實回報、勝率、Sharpe Ratio |
| **Win Rate 修正** | 模擬結束時若仍持有股票，會計算未實現盈虧並計入勝率 |
| **風險指標** | 新增 Sharpe Ratio、Sortino Ratio、利潤因子、平均持倉天數 |
| **基準對比** | 新增買入持有 (Buy & Hold) 基準，顯示策略超額報酬 (Alpha) |
| **預測準確度儀表板** | 顯示各股票各時間範圍的真實預測準確度，含長條圖 |
| **組合模擬** | 多股票組合模擬，資金平均分配，顯示組合表現 |
| **信心度加權策略** | 根據信號信心度調整倉位大小 (30%-100%)，高信心=大倉位 |
| **蒙地卡羅測試** | 隨機翻轉信號 1000 次，顯示報酬分佈、獲利機率、風險指標 |
| **yfinance 優先** | 資料來源改為 yfinance 優先，akshare 作為備援 |

### 修改的檔案
- `src/simulator.py` — 新增風險指標、基準對比、組合模擬、信心度加權、蒙地卡羅
- `src/model_monitoring.py` — 重寫 calculate_accuracy 使用真實價格驗證、重寫回測引擎
- `app/pages/1_💰_投資模擬器.py` — 新增風險指標顯示、基準對比、準確度分析、進階功能
- `src/data_fetcher.py` — yfinance 優先，akshare 備援

### 改進項目 (2026-09-04)

| 改進 | 說明 |
|---|---|
| **投資模擬器** | 新增 Streamlit 模擬頁面，可自訂日期範圍、資金、時間範圍，模擬跟單收益 |
| **交易成本** | 模擬包含佣金 0.1% + 印花稅 0.13%，計算淨收益 |
| **多頁面導航** | Streamlit 側邊欄自動顯示頁面導航 (預測儀表板 + 投資模擬器) |

### 修改的檔案
- `src/simulator.py` — 新增投資模擬引擎
- `app/pages/1_💰_投資模擬器.py` — 新增投資模擬互動頁面

### 改進項目 (2026-09-03)

| 改進 | 說明 |
|---|---|
| **日誌系統** | 全面使用 `setup_logger()` 取代 `print()`，統一日誌格式 |
| **單例模式** | Supabase 客戶端改為單例模式，避免重複連接 |
| **勝率計算修正** | `get_win_rate()` 改為驗證實際價格變動 (之前返回虛假數據) |
| **MFI 向量化** | MFI 計算從 Python 迴圈改為 NumPy 向量化運算 (速度提升 ~10x) |
| **環境變數檢查** | 啟動時自動檢查 `.env` 檔案是否存在，缺失時提供清楚錯誤訊息 |
| **依賴版本鎖定** | requirements.txt 加入版本上限 (`<2.0.0`)，防止未來不相容更新 |
| **型別標註** | cleanup_old.py, predict_upload.py 新增型別提示，提升程式碼可維護性 |

### 修改的檔案
- `config.py` — 新增 `.env` 檔案存在檢查
- `src/cleanup_old.py` — 使用 logger、型別標註、新增刪除前記錄筆數顯示
- `src/predict_upload.py` — 單例 Supabase 客戶端、修正勝率計算、型別標註
- `src/feature_engineering.py` — MFI 計算向量化
- `src/train_model.py` — 新增文件字串
- `requirements.txt` — 版本鎖定、分類整理

## 授權

本專案僅供學習和研究使用，不構成任何投資建議。投資有風險，入市需謹慎。
