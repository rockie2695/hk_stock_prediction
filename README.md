# 港股每日自動預測系統

純機器學習的港股每日預測系統，使用 XGBoost / LightGBM 進行漲跌預測，結果自動上傳至 Supabase 雲端資料庫，並透過 Streamlit 網站展示。

## 系統架構

```
Windows 本地定時訓練 → 預測結果上傳至 Supabase (PostgreSQL) → Streamlit 網站顯示
```

## 功能特色

- **多時間範圍預測**: 同時預測明日(1天)、下週(5天)、下月(20天)
- **自動模型選擇**: Optuna 自動調參，XGBoost vs LightGBM 自動選出冠軍
- **27項技術指標**: 包含市場指數、匯率、統計特徵
- **模型指標追蹤**: 記錄 F1 Score、AUC Score、冠軍模型類型
- **互動式儀表板**: Streamlit 顯示預測結果、信心度趨勢、信號分佈
- **風險管理**: 止損/止盈建議、風險報酬比、預期報酬
- **信心度追蹤**: 顯示信心度變化趨勢 (↑↓→)
- **勝率統計**: 歷史預測準確率追蹤

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
- 各時間範圍的冠軍模型 (xgboost/lightgbm)
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
├── test_quick.py         # 快速測試腳本
├── logs/                 # 日誌資料夾
├── models/               # 訓練好的模型 (.pkl)
├── src/
│   ├── __init__.py
│   ├── logger.py         # 日誌設定
│   ├── init_database.py  # 自動建表 (冪等)
│   ├── data_fetcher.py   # 下載港股歷史數據 (akshare/yfinance)
│   ├── feature_engineering.py  # 27項技術指標計算
│   ├── train_model.py    # Optuna 自動調參 + Walk-Forward 驗證
│   └── predict_upload.py # 每日預測並上傳 Supabase
└── app/
    ├── __init__.py
    └── streamlit_app.py  # Streamlit 預測儀表板
```

## 技術細節

### 機器學習模型
- **演算法**: XGBoost / LightGBM
- **超參數優化**: Optuna (50 trials per model)
- **交叉驗證**: TimeSeriesSplit (n_splits=5)
- **模型選擇**: 基於 F1 Score 自動選出冠軍
- **訓練數據**: 3 年歷史數據 (約 750 交易日)
- **類別權重**: 自動平衡正負樣本 (上限3倍)，避免模型偏向多數類

### 技術指標 (27 Features)

| 類別 | 特徵 | 說明 |
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

### 目標變數 (Target)
- **目標**: N天後收盤價 > 今日收盤價 → 1 (Buy)，否則 → 0
- **類別權重**: 自動平衡正負樣本 (上限3倍)
- **多時間範圍**: 1天、5天、20天

### 模型表現 (F1 Score)
| 時間範圍 | F1 Score | 說明 |
|---|---|---|
| 1天 | ~0.57 | 可用 — 短期趨勢 |
| 5天 | ~0.69 | 良好 — 中期動量 |
| 20天 | ~0.73 | 最佳 — 長期趨勢 |

**注意**: 股票預測本身非常困難，AUC ~0.55-0.60 已是合理範圍。

### 信號判定
| 信心度 | 信號 |
|---|---|
| > 55% | Buy (買入) |
| < 45% | Sell (賣出) |
| 45% ~ 55% | Hold (持有) |

### 資料來源
- **股票數據**: akshare (主) / yfinance (備)
- **市場指數**: yfinance (^HSI 恒生指數)
- **匯率**: yfinance (USD/HKD)

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
    model_type TEXT,          -- 'xgboost' 或 'lightgbm'
    f1_score FLOAT8,          -- 模型 F1 分數
    auc_score FLOAT8,         -- 模型 AUC 分數
    expected_return FLOAT8,   -- 預期報酬率 (%)
    risk_reward FLOAT8,       -- 風險報酬比
    stop_loss FLOAT8,         -- 止損點 (%)
    take_profit FLOAT8,       -- 止盈點 (%)
    confidence_trend TEXT,    -- 信心度趨勢: ↑↓→-
    win_rate FLOAT8,          -- 歷史勝率 (%)
    created_at TIMESTAMPTZ DEFAULT NOW()
);
```

## 注意事項

- 所有日期時間使用香港時區 (`Asia/Hong_Kong`)
- 特徵計算嚴禁 Look-ahead Bias（只使用當天之前的數據）
- `.env` 檔案包含敏感資訊，請勿上傳至版本控制
- 模型檔案 (`.pkl`) 不上傳至版本控制
- 可重複執行 `train_model.py` 和 `predict_upload.py`，會自動覆蓋

## 常見問題

### Q: 為什麼 AUC 只有 0.55-0.60？
A: 股票預測本身非常困難。即使是大型對沖基金，AUC 也通常在 0.55-0.65 之間。你的模型已達到合理範圍。

### Q: F1 Score 代表什麼？
A: F1 = 精準率與召回率的平衡。F1 > 0.5 表示模型比隨機好，F1 > 0.6 表示可用於交易信號。

### Q: 可以添加更多股票嗎？
A: 修改 `.env` 中的 `STOCK_LIST`，例如 `STOCK_LIST=0700,9988,0005,0939,1810`

### Q: 如何查看訓練日誌？
A: 日誌位於 `logs/app.log`

### Q: 訓練要多久？
A: 約 5-10 分鐘 (取決於股票數量和 Optuna trials)

### Q: 模型會自動更新嗎？
A: 需要手動執行 `train_model.py` 重新訓練，或設定 Windows 排程器自動執行
