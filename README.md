# 港股每日自動預測系統

純機器學習的港股每日預測系統，使用 XGBoost / LightGBM 進行漲跌預測，結果自動上傳至 Supabase 雲端資料庫，並透過 Streamlit 網站展示。

## 系統架構

```
Windows 本地定時訓練 → 預測結果上傳至 Supabase (PostgreSQL) → Streamlit 網站顯示
```

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
SUPABASE_DB_PASSWORD=your-db-password
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
├── logs/                 # 日誌資料夾
├── models/               # 訓練好的模型 (.pkl)
├── src/
│   ├── __init__.py
│   ├── logger.py         # 日誌設定
│   ├── init_database.py  # 自動建表 (冪等)
│   ├── data_fetcher.py   # 下載港股歷史數據
│   ├── feature_engineering.py  # 技術指標計算
│   ├── train_model.py    # Optuna 自動調參 + Walk-Forward 驗證
│   └── predict_upload.py # 每日預測並上傳 Supabase
└── app/
    ├── __init__.py
    └── streamlit_app.py  # Streamlit 預測儀表板
```

## 技術細節

- **機器學習模型**: XGBoost / LightGBM，使用 Optuna 自動調參
- **交叉驗證**: TimeSeriesSplit (n_splits=5)，嚴禁隨機 Shuffle
- **特徵工程**: RSI, MACD, 布林通道, ATR, 收益率, 成交量比率
- **目標變數**: 明日收盤價 > 今日收盤價 → 1 (Buy), 否則 → 0 (Sell)
- **信號判定**: Buy (>0.55), Sell (<0.45), Hold (0.45-0.55)

## 注意事項

- 所有日期時間使用香港時區 (`Asia/Hong_Kong`)
- 特徵計算嚴禁 Look-ahead Bias（只使用當天之前的數據）
- `.env` 檔案包含敏感資訊，請勿上傳至版本控制
- 模型檔案 (`.pkl`) 不上傳至版本控制
