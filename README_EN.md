# Hong Kong Stock Daily Auto-Prediction System

[中文](README.md)

Pure ML-based daily HK stock prediction system using XGBoost / LightGBM / RandomForest / CatBoost ensemble models for up/down prediction. Results are auto-uploaded to Supabase cloud database and displayed via Streamlit dashboard.

> **Warning:** This system is for educational and research purposes only. It does not constitute investment advice. Investing involves risk — invest with caution.

## System Architecture

```
Local Windows scheduled training (parallel) → Upload predictions to Supabase (PostgreSQL) → Streamlit website display
```

## Features

### Core Features
- **Multi-timeframe Prediction**: Predicts tomorrow (1-day), next week (5-day), next month (20-day) simultaneously
- **Four-model Ensemble**: XGBoost + LightGBM + RandomForest + CatBoost
- **Three Ensemble Modes**: Voting (weighted average) / Stacking (meta-model) / Blending (out-of-fold)
- **Parallel Training**: Timeframes trained simultaneously, ~3x speedup
- **Parallel Prediction**: Multiple stocks predicted concurrently
- **SMOTE Class Balancing**: Automatically handles positive/negative class imbalance
- **33 Technical Indicators**: Includes momentum, volatility, Williams %R, MFI, etc.
- **15 Extended Features**: Sentiment analysis, sector rotation, short selling, connect flow, market regime detection
- **Feature Correlation Filter**: Auto-removes redundant features with |corr| > 0.9
- **Threshold Optimization**: Auto-searches optimal Buy/Sell confidence thresholds (replaces fixed 0.55/0.45)
- **Model Versioning**: Timestamped model backups, keeps latest 5 versions, supports rollback
- **Model Metrics Tracking**: Records F1 Score, AUC Score, champion model type
- **Model Disagreement Detection**: Forces Hold when model disagreement >= 50%, displays disagreement level
- **GPU Support**: CatBoost can optionally use GPU acceleration (enable via `USE_GPU=True`)
- **Interactive Dashboard**: Streamlit displays prediction results, confidence trends, signal distribution
- **K-line Chart (with Buy/Sell Signals)**: Interactive K-line chart marking model-predicted buy/sell signal positions, supports custom stock and timeframe selection
- **MA Overlay**: K-line chart can overlay MA5 (short-term), MA10 (mid-term), MA20 (long-term) moving averages
- **Stock Comparison Chart**: Select two stocks to compare cumulative return trends and performance metrics side-by-side
- **Technical Indicator Display**: Shows latest RSI, MACD, Stochastic, ADX, MFI, Bollinger Band Width, ATR, Volume Ratio values as signal basis
- **Signal Confirmation Analysis**: Each Buy/Sell signal auto-analyzes indicator alignment (support/contradict/neutral) to help assess signal reliability
- **Threshold Interactive Control**: Charts can show Buy/Sell threshold lines for selected timeframes, avoiding line overlap
- **Data Export**: Supports CSV and Excel format export of prediction records
- **Investment Simulator**: Simulates investment based on Buy/Sell signals, calculates actual returns, win rate, max drawdown, supports custom capital and date range
- **Risk Metrics**: Sharpe Ratio, Sortino Ratio, Profit Factor, Average Holding Days
- **Benchmark Comparison**: Strategy return vs Buy & Hold, displays excess return (Alpha)
- **Prediction Accuracy Analysis**: Validates whether historical predictions were correct, shows true accuracy per stock and timeframe
- **Portfolio Simulation**: Multi-stock portfolio simulation, shows portfolio performance and diversification effects
- **Confidence-Weighted Strategy**: Adjusts position size based on signal confidence (high confidence = larger position)
- **Monte Carlo Test**: Randomly flips signals to test strategy robustness, shows return distribution (customizable 100-5000 simulations)
- **HK Public Holiday Calendar**: Prediction dates auto-skip HK market holidays, fetches holiday data from official 1823.gov.hk API (including Lunar New Year)
- **Slippage Simulation**: Simulates real trading slippage — buy price slightly higher, sell price slightly lower (adjustable 0-1%)
- **Board Lot Trading**: Calculates buy quantity per HK minimum trading unit (board lot), closer to real trading
- **Stop Loss/Take Profit Execution**: Auto-closes positions based on predicted stop loss/take profit levels (checks daily prices between signal dates)
- **Local Data Cache**: Uses Parquet format for caching historical data — 4-hour cache on trading days, 24-hour cache on non-trading days

### Risk Management
- **Stop Loss/Take Profit Recommendations**: Auto-calculates recommended stop loss and take profit levels based on volatility
- **Risk-Reward Ratio**: Evaluates the ratio of potential gain to risk
- **Expected Return**: Estimates expected return rate based on confidence and volatility
- **Confidence Tracking**: Shows confidence change trend (↑↓→)
- **Win Rate Statistics**: Historical prediction accuracy tracking
- **Model Disagreement Detection**: Auto-Hold when models disagree, avoiding weak decisions

### Model Monitoring
- **Data Quality Checks**: Auto-detects missing dates, abnormal confidence distribution
- **True Accuracy Verification**: Validates predictions using actual price data (not just confidence as proxy)
- **Rolling Accuracy Tracking**: 30-day rolling window for true prediction accuracy (Did price rise after Buy? Fall after Sell?)
- **Training Metrics Display**: Shows F1 Score, AUC Score, champion model type for each stock and timeframe
- **Model Drift Detection**: Monitors model performance degradation (>10% = high, >5% = moderate)
- **Backtest Engine**: Actually simulates strategy performance, calculates real returns, win rate, Sharpe Ratio
- **Signal Alerts**: Auto-alerts for strong signals (confidence >70% or expected return >5%)
- **Confidence Calibration**: Ensures confidence scores are reliable

## Quick Start

### 1. Install Dependencies

```bash
# Method One: One-click install via batch file
setup.bat

# Method Two: Manual installation
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure Environment Variables

1. Copy `.env.example` to `.env`
2. Fill in your Supabase project info:

```env
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-anon-key
STOCK_LIST=0700,9988,0005,0939

# Model training toggles
USE_ENSEMBLE=True
USE_STACKING=False
USE_BLENDING=False
USE_CATBOOST=True
USE_SMOTE=True
USE_GPU=False

# Extended feature toggles
USE_SENTIMENT=True
USE_SECTOR=True
USE_SHORT_SELL=True
USE_CONNECT=True
USE_REGIME=True
USE_ONLINE_LEARNING=False
USE_DYNAMIC_WEIGHTING=False
```

**⚠️ IMPORTANT:** Obtain your project URL and key from the [Supabase website](https://supabase.com), fill in `.env`, then proceed.

> **Warning:** NEVER commit `.env` files to version control (Git). This file contains sensitive info like Supabase keys. Verify `.env` is in `.gitignore`.

### 3. Initialize Database

```bash
python src/init_database.py
```

### 4. Train Model

```bash
python src/train_model.py
```

After training completes, you will see:
- Champion model per timeframe (xgboost/lightgbm/catboost/voting/stacking/blending)
- Model comparison table (F1 scores)
- F1 Score and AUC Score
- Top 10 feature importance

### 5. Daily Prediction & Upload

```bash
python src/predict_upload.py
```

### 6. Launch Prediction Dashboard

```bash
streamlit run app/streamlit_app.py
```

Dashboard contains two pages (sidebar toggle):
- **📈 Prediction Dashboard**: Tabbed design, includes:
  - **📊 Signal Overview**: Signal cards (1d/5d/20d) + indicator alignment analysis + signal distribution
  - **📈 K-line & Indicators**: K-line chart (with buy/sell signals) + technical indicators + stock comparison
  - **🎯 Confidence Trend**: Confidence change chart + Buy/Sell threshold lines
  - **📋 Prediction Records**: Recent prediction table + export functionality
  - **🔍 Model Performance**: Rolling accuracy + training metrics + model monitoring + market regime
- **💰 Investment Simulator**: Custom date range, capital, timeframe, simulates copy-trading returns
- **📊 Strategy Backtest**: Equity curve, drawdown, trade log, vs buy-and-hold
- **💼 Portfolio**: Holdings, signal/confidence distribution, risk exposure

### 7. Run Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_config.py -v

# Run specific test class
python -m pytest tests/test_feature_engineering.py::TestFeatureEngineering -v

# Run specific test function
python -m pytest tests/test_train_model.py::TestModelTraining::test_train_xgboost -v

# Show verbose output
python -m pytest tests/ -v --tb=long

# Show only failed tests
python -m pytest tests/ -v --tb=short
```

**Test Coverage:**
| Test File | Tests | Coverage |
|---|---|---|
| `test_config.py` | 10 | Env vars, stock list parsing, Supabase config |
| `test_feature_engineering.py` | 17 | RSI, MACD, Bollinger, ATR, ADX, Stochastic, MFI, Williams %R |
| `test_train_model.py` | 14 | XGBoost, LightGBM, RandomForest, CatBoost (incl. early stopping), SMOTE, Blending |
| `test_predict.py` | 10 | Prediction dates, model loading, signal determination, upload |
| `test_sentiment.py` | 4 | News sentiment feature computation |
| `test_sector.py` | 4 | Sector rotation feature computation |
| `test_short_selling.py` | 4 | Short selling feature computation |
| `test_connect_flow.py` | 4 | Connect flow feature computation |
| `test_regime.py` | 4 | Market regime detection |
| `test_online_learner.py` | 3 | Online learning |
| `test_dynamic_weighting.py` | 5 | Dynamic ensemble weighting |
| `test_backtest.py` | 2 | Backtest page |
| `test_portfolio.py` | 2 | Portfolio page |
| **Total** | **85** | |

## Windows Task Scheduler Setup

Set up Windows Task Scheduler to auto-execute daily at 16:30 (after HK market close):

1. Press `Win + R`, enter `taskschd.msc` to open Task Scheduler
2. Click "Create Basic Task..." on the right
3. Name: `HK Stock Daily Prediction`
4. Trigger: Select "Daily", set start time to `16:30`
5. Action: Select "Start a program"
6. Program or script: Browse and select `run_daily.bat`
7. After completion, right-click the task → Properties → Settings:
   - ✅ Wake computer to run this task
   - ✅ Run whether user is logged on or not

> **Note:** HK market trading hours are 9:30-16:00 HKT. Schedule at 16:30 to ensure closing data is fully loaded. Weekends and public holidays are non-trading days — the system auto-skips them.

## Docker Deployment

### Build Image

```bash
docker build -t hk-stock-prediction .
```

### Run Dashboard

```bash
docker run -d -p 8501:8501 --env-file .env --name stock-dashboard hk-stock-prediction
```

Open browser and visit `http://localhost:8501`

### Run Training (one-time)

```bash
docker run --env-file .env hk-stock-prediction python src/train_model.py
```

### Run Daily Prediction

```bash
docker run --env-file .env hk-stock-prediction python src/predict_upload.py
```

### Windows Task Scheduler (Docker)

Use Windows Task Scheduler to auto-run Docker container daily at 16:30:

```bash
docker run --env-file C:\hk_stock_prediction\.env hk-stock-prediction python src/predict_upload.py
```

### Parameter Description

| Parameter | Description |
|---|---|
| `-d` | Run in background |
| `-p 8501:8501` | Maps Streamlit default port |
| `--env-file .env` | Load environment variables |
| `--name stock-dashboard` | Container name |

> **Warning:** NEVER hardcode Supabase keys or any `.env` content inside Docker images. Use `--env-file` to inject environment variables at runtime. Never COPY `.env` files into the image.

> **Note:** When using GPU acceleration (CatBoost), Docker requires NVIDIA Container Toolkit. Ensure the host machine has NVIDIA drivers installed and run the container with `--gpus all`.

## Project Structure

```
project_root/
├── .env.example          # Environment variable template
├── .gitignore            # Git ignore list
├── .dockerignore         # Docker ignore list
├── Dockerfile            # Docker image definition
├── requirements.txt      # Python dependencies
├── config.py             # Reads .env, provides global config
├── run_daily.bat         # Windows batch file (for scheduler)
├── setup.bat             # One-click dependency install
├── logs/                 # Log folder
├── models/               # Trained models (.pkl)
│   ├── best_model_{tf}.pkl           # Current model
│   ├── best_model_{tf}_{ts}.pkl      # Versioned models (keeps latest 5)
│   ├── feature_importance_{tf}.csv   # Feature importance
│   └── roc_curve_{tf}.png            # ROC curve
├── cache/                # Local data cache
├── tests/                # Unit tests
│   ├── __init__.py
│   ├── test_config.py           # Config module tests
│   ├── test_feature_engineering.py  # Feature engineering tests
│   ├── test_train_model.py      # Model training tests
│   └── test_predict.py          # Prediction upload tests
├── src/
│   ├── __init__.py
│   ├── logger.py         # Logger configuration
│   ├── init_database.py  # Auto-create tables (idempotent)
│   ├── data_fetcher.py   # Download HK stock data (akshare/yfinance)
│   ├── feature_engineering.py  # 48 technical indicators (33 base + 15 extended)
│   ├── train_model.py    # Optuna tuning + Voting/Stacking ensemble + SMOTE
│   ├── predict_upload.py # Daily prediction & upload to Supabase
│   ├── simulator.py      # Investment simulator (signal simulation, costs, performance tracking)
│   ├── cleanup_old.py    # Cleanup old data (keeps 60 days)
│   ├── model_monitoring.py  # Data quality, model drift, alerts, calibration
│   ├── sentiment.py      # News sentiment features
│   ├── sector.py         # Sector rotation features
│   ├── short_selling.py  # Short selling features
│   ├── connect_flow.py   # Connect flow features
│   ├── regime.py         # Market regime detection (bull/bear/sideways)
│   ├── online_learner.py # Online learning (warm-start)
│   └── dynamic_weighting.py  # Dynamic ensemble weighting
├── app/
│   ├── __init__.py
│   ├── streamlit_app.py  # Streamlit prediction dashboard
│   └── pages/
│       ├── __init__.py
│       ├── 1_💰_投資模擬器.py  # Investment simulator page
│       ├── 2_backtest.py      # Strategy backtest page
│       └── 3_portfolio.py     # Portfolio page
├── migrate_metrics.sql   # DB migration: model metrics fields
├── migrate_quick_wins.sql # DB migration: risk management fields
├── migrate_thresholds.sql # DB migration: Buy/Sell threshold fields
├── migrate_disagreement.sql # DB migration: model disagreement metrics
└── migrations/
    └── 003_extended_features.sql # Extended features docs (no DB changes)
```

> **Warning:** `.env` and `.pkl` files must NOT be committed to version control. `.gitignore` is configured to ignore them. If accidentally committed, remove them from Git history immediately.

## Technical Details

### Machine Learning Models
- **Algorithms**: XGBoost + LightGBM + RandomForest + CatBoost ensemble
- **Ensemble Methods**: VotingClassifier (soft voting) or StackingClassifier (meta-model = LogisticRegression) or Blending (out-of-fold)
- **Hyperparameter Optimization**: Optuna (50 trials, simultaneously searching 4 models + voting weights)
- **Weight Optimization**: Optuna auto-searches optimal weight combination (e.g. [0.3, 0.3, 0.2, 0.2]), not fixed 1:1:1:1
- **Cross-Validation**: TimeSeriesSplit (n_splits=5), strictly follows time order, no future data leakage
- **Class Imbalance Handling**: SMOTE (applied only on training folds, never across validation folds)
- **Training Data**: 3 years of historical data (~750 trading days)
- **Evaluation Metrics**: F1 Score, AUC, Precision, Recall
- **ROC Curve**: Auto-saved to `models/roc_curve_{timeframe}.png`
- **Feature Correlation Filter**: Auto-removes redundant features with |corr| > 0.9
- **Threshold Optimization**: Auto-searches optimal Buy/Sell confidence thresholds (replaces fixed 0.55/0.45)
- **Model Versioning**: Timestamped backups, auto-keeps latest 5 versions
- **Feature Importance**: Output to `models/feature_importance_{timeframe}.csv`
- **Model Disagreement**: Forces Hold when 4 models disagree >= 50%
- **Feature Alignment**: Saves `feature_columns` during training, strictly uses same order during prediction
- **CatBoost Early Stopping**: Uses early_stopping_rounds=30 with eval_set validation, auto-stops training to prevent overfitting (iterations 200-300)
- **GPU Support**: CatBoost can optionally use GPU acceleration (enable via `USE_GPU=True`)

> **Note:** Feature engineering MUST NOT contain Look-ahead Bias. All technical indicators are computed using only data from the current day and earlier — never future data. Any feature calculation violation will cause inflated model performance that fails in real prediction.

> **Warning:** SMOTE is applied ONLY on training folds of TimeSeriesSplit, never across validation folds. This is critical to prevent data leakage. If SMOTE is applied on the entire training set, the model will achieve inflated AUC on validation.

> **Note:** GPU acceleration requires an NVIDIA GPU and installed NVIDIA drivers. CatBoost GPU training uses more memory — at least 4GB VRAM recommended. Non-NVIDIA GPUs (e.g., AMD) are not supported.

### Technical Indicators (48 Features)

**Base Technical Indicators (33 Features)：**

| Category | Feature | Description |
|---|---|---|
| **Returns** | `ret_1d`, `ret_3d`, `ret_5d`, `ret_10d`, `ret_20d`, `ret_30d` | 1/3/5/10/20/30-day price change |
| **Price Pattern** | `high_low_range`, `close_to_high`, `close_to_low` | Intraday range, closing position |
| **Price Position** | `ma50_deviation` | Current price deviation from 50-day MA |
| **Volume** | `vol_ratio_5d`, `vol_ratio_10d` | Relative volume strength |
| **Volume** | `obv_change` | OBV (On-Balance Volume) change |
| **Volume** | `volume_cv` | Volume coefficient of variation (20-day) |
| **Momentum** | `rsi_14` | RSI overbought/oversold |
| **Momentum** | `stoch_k`, `stoch_d` | Stochastic oscillator |
| **Momentum** | `mfi` | Money Flow Index |
| **Momentum** | `williams_r` | Williams %R |
| **Trend** | `macd_diff`, `macd_dea`, `macd_hist` | MACD three components |
| **Trend** | `adx` | Trend strength (direction-agnostic) |
| **Volatility** | `bb_width` | Bollinger Band width |
| **Volatility** | `atr_14`, `atr_ratio` | Average True Range, ATR/close ratio |
| **Statistics** | `ret_5d_skew`, `ret_5d_kurt` | Return skewness/kurtosis |
| **Statistics** | `volatility_10d`, `volatility_20d` | 10-day/20-day volatility |
| **Market** | `hsi_ret_5d`, `hsi_ret_20d` | Hang Seng Index return |
| **FX** | `usdhkd_change` | USD/HKD exchange rate change |

**Extended Features (15 Features)：**

| Category | Feature | Description |
|---|---|---|
| **Sentiment** | `sentiment_5d`, `sentiment_10d`, `sentiment_change` | 5-day/10-day sentiment score and change (news/social) |
| **Sector** | `sector_momentum_5d`, `sector_momentum_20d`, `sector_vs_hsi` | Sector momentum and relative HSI performance |
| **Short Selling** | `short_sell_ratio`, `short_sell_ratio_5d`, `short_sell_ratio_change` | Real-time/5-day short selling ratio and change |
| **Connect Flow** | `southbound_net_5d`, `southbound_momentum`, `connect_sentiment` | Southbound net inflow, momentum, sentiment |
| **Regime** | `market_regime`, `regime_confidence`, `hsi_trend_50_200` | Bull/Bear/Sideways regime, confidence, MA ratio |

### Model Training Toggles

| Env Var | Default | Description |
|---|---|---|
| `USE_ENSEMBLE` | `True` | Enable ensemble (False = single model comparison) |
| `USE_STACKING` | `False` | Use StackingClassifier (meta-model learns combination) |
| `USE_BLENDING` | `False` | Use Blending (out-of-fold stacking, usually more accurate) |
| `USE_CATBOOST` | `True` | Include CatBoost as 4th model |
| `USE_SMOTE` | `True` | Enable SMOTE class imbalance handling |
| `USE_GPU` | `False` | CatBoost GPU training (requires NVIDIA GPU, uses more memory) |

**Priority Rules：**
- `USE_STACKING=True` or `USE_BLENDING=True` → Forces ensemble mode
- `USE_ENSEMBLE=True` (default) → VotingClassifier (weighted average)
- `USE_ENSEMBLE=False` → Single best model (XGBoost vs LightGBM vs CatBoost)

**Training Speed：**
- Timeframes (1d, 5d, 20d) **trained in parallel**, ~3x speedup
- Multi-stock prediction also supports **parallel processing**

### Target Variable
- **Target**: N-day closing price > today's closing price → 1 (Buy), otherwise → 0
- **Class Weights**: Auto-balances positive/negative samples (max 3x) + SMOTE augmentation
- **Multi-timeframe**: 1-day, 5-day, 20-day

### Model Performance (F1 Score)
| Timeframe | F1 Score | Description |
|---|---|---|
| 1-day | ~0.57 | Usable — short-term trend |
| 5-day | ~0.69 | Good — mid-term momentum |
| 20-day | ~0.73 | Best — long-term trend |

**Note**: Stock prediction is inherently difficult. An AUC of 0.55-0.60 is a reasonable range.

> **Note:** Stock prediction is inherently difficult. Even large hedge funds typically achieve AUC between 0.55-0.65. An AUC of 0.55-0.60 indicates some predictive ability, not failure.

> **Warning:** Past performance does NOT guarantee future results. Model F1/AUC on historical data does not guarantee future returns. Never equate model performance with trading guarantees. When using the investment simulator, always consider trading costs and slippage impact.

### Signal Determination
- Thresholds are auto-optimized by the model (searches for best F1 on validation set). Defaults below:
| Confidence | Signal |
|---|---|
| > threshold (optimized, default 55%) | Buy |
| < threshold (optimized, default 45%) | Sell |
| Otherwise | Hold |

### Risk Management Metrics

| Metric | Description | Calculation |
|---|---|---|
| **Expected Return** | Based on confidence & volatility | `(confidence-0.5) × 2 × volatility × √days` |
| **Stop Loss** | Recommended stop loss level | `2 × volatility × √days` |
| **Take Profit** | Recommended take profit level | `1.5 × \|expected return\|` |
| **Risk-Reward** | Gain to risk ratio | `return / risk` |
| **Confidence Trend** | Direction of change | ↑up ↓down →flat |
| **Win Rate** | Historical accuracy | `Buy+Sell signal ratio` |

### Data Sources
- **Stock Data**: yfinance (primary) / akshare (fallback)
- **Market Index**: yfinance (^HSI Hang Seng Index)
- **Exchange Rate**: yfinance (USD/HKD)

### Investment Simulation

Simulates copy-trading based on historical prediction signals, calculates actual investment returns.

| Item | Description |
|---|---|
| **Initial Capital** | HKD 20,000 per stock (customizable on page) |
| **Buy Signal** | Buy max whole shares at closing price |
| **Sell Signal** | Sell all shares at closing price |
| **Hold Signal** | No trade (counted as win/loss at end) |
| **No Short Selling** | Sell only to close position, no shorting |

**Trading Costs (HK Standard)：**

| Fee | Rate | Description |
|---|---|---|
| Commission | 0.1% | Minimum HKD 20 per trade |
| Stamp Duty | 0.13% | Sell-side only |

> **Note:** Trading costs are based on HK standards: 0.1% commission (min HKD 20) + 0.13% stamp duty (sell-side only). Rates may vary by broker — verify with your broker. Trading costs vary greatly across markets; do not directly apply to other markets.

**Calculation Formulas：**
```
Buy shares = floor(investable amount / (price × (1 + commission rate)))

Sell proceeds = shares × price × (1 - commission rate - stamp duty rate)

P/L = sell proceeds - buy cost
```

**Output Metrics：**

| Metric | Description |
|---|---|
| Total P/L | Final portfolio value - initial capital |
| Return Rate | (Final - Initial) / Initial × 100 |
| Trade Count | Buy + sell count |
| Win Rate | Winning trades / total trades × 100% (incl. positions held to expiry) |
| Max Drawdown | Percentage decline from peak to trough |
| Sharpe Ratio | Annualized risk-adjusted return (>1 good, >2 great) |
| Sortino Ratio | Risk-adjusted return considering only downside risk |
| Profit Factor | Total profit / total loss (>1 means profit > loss) |
| Avg Holding Days | Average days from buy to sell |
| Buy & Hold Return | Baseline: buy at start and hold to end |
| Excess Return (Alpha) | Strategy return - buy & hold return |

**Advanced Features：**

| Feature | Description |
|---|---|
| **Portfolio Simulation** | Simulates portfolio of all selected stocks, equal capital allocation, shows portfolio Sharpe and max drawdown |
| **Confidence-Weighted** | Adjusts position size by signal confidence (30%-100% capital), high confidence = larger position |
| **Monte Carlo Test** | Randomly flips signals 1000 times, tests robustness, shows return distribution and profit probability |
| **Prediction Accuracy** | Validates historical predictions: Did price rise after Buy? Fall after Sell? |
| **Best Timing Mode** | Assumes foresight of N-day prices, picks best buy/sell days (hindsight mode) |

## Database Schema

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
    f1_score FLOAT8,          -- Model F1 score
    auc_score FLOAT8,         -- Model AUC score
    expected_return FLOAT8,   -- Expected return (%)
    risk_reward FLOAT8,       -- Risk-reward ratio
    stop_loss FLOAT8,         -- Stop loss (%)
    take_profit FLOAT8,       -- Take profit (%)
    confidence_trend TEXT,    -- Confidence trend
    win_rate FLOAT8,          -- Historical win rate (%)
    threshold_buy FLOAT8,     -- Optimized Buy threshold (varies per timeframe)
    threshold_sell FLOAT8,    -- Optimized Sell threshold (varies per timeframe)
    model_disagreement FLOAT8, -- Model disagreement (0=agree, 0.5=split, 1=full disagreement)
    model_split TEXT,          -- Model vote results (e.g., '4/0', '3/1', '2/2')
    created_at TIMESTAMPTZ DEFAULT NOW()
);
```

### Database Migrations

Execute the following SQL to add new columns:

```sql
-- Model metrics fields
ALTER TABLE stock_predictions ADD COLUMN IF NOT EXISTS model_type TEXT;
ALTER TABLE stock_predictions ADD COLUMN IF NOT EXISTS f1_score FLOAT8;
ALTER TABLE stock_predictions ADD COLUMN IF NOT EXISTS auc_score FLOAT8;

-- Risk management fields
ALTER TABLE stock_predictions ADD COLUMN IF NOT EXISTS expected_return FLOAT8;
ALTER TABLE stock_predictions ADD COLUMN IF NOT EXISTS risk_reward FLOAT8;
ALTER TABLE stock_predictions ADD COLUMN IF NOT EXISTS stop_loss FLOAT8;
ALTER TABLE stock_predictions ADD COLUMN IF NOT EXISTS take_profit FLOAT8;
ALTER TABLE stock_predictions ADD COLUMN IF NOT EXISTS confidence_trend TEXT DEFAULT '-';
ALTER TABLE stock_predictions ADD COLUMN IF NOT EXISTS win_rate FLOAT8;

-- Model disagreement metrics
ALTER TABLE stock_predictions ADD COLUMN IF NOT EXISTS model_disagreement FLOAT8 DEFAULT 0;
ALTER TABLE stock_predictions ADD COLUMN IF NOT EXISTS model_split TEXT DEFAULT '0/0';

-- Remove unique constraint (keep history)
ALTER TABLE stock_predictions DROP CONSTRAINT IF EXISTS unique_stock_prediction;
```

## Model Monitoring

### Data Quality Checks
- Detect missing dates (minimum 5 predictions per timeframe)
- Check confidence distribution is reasonable
- Detect abnormal signal distribution (single signal >70%)

### True Accuracy Verification
- Fetch actual price data via yfinance
- Buy signal: N-day closing > prediction day closing = correct
- Sell signal: N-day closing < prediction day closing = correct
- Display true accuracy percentage per stock and timeframe

### Model Drift Detection
- Compare recent 14-day vs 60-day prediction accuracy (verified with actual prices)
- Accuracy drop >10% = high alert
- Accuracy drop >5% = moderate alert

### Backtest Engine
- Actually simulates strategy: buy → sell → calculate real returns
- Calculates Sharpe Ratio, win rate, max drawdown
- Compares to buy & hold baseline, calculates excess return (Alpha)

### Signal Alerts
- Strong signals with confidence >70%
- High-return signals with expected return >5%

### Confidence Calibration
- Monitors whether average confidence is reasonable
- Over-confidence (>60%) or under-confidence (<40%) triggers adjustment suggestions

## Important Notes

- All dates/times use Hong Kong timezone (`Asia/Hong_Kong`)
- Feature computation MUST NOT contain Look-ahead Bias (only use data before the current day)
- `.env` contains sensitive info — do not upload to version control
- Model files (`.pkl`) must not be uploaded to version control
- Daily predictions keep history (auto-clean data older than 60 days)

> **Warning:** Feature engineering MUST NOT contain Look-ahead Bias. This is one of the most common mistakes in quantitative trading. All features must be computed using only data from the current day and earlier, ensuring the model can replicate the same performance in live trading.

> **Warning:** `.env` files contain Supabase keys and database credentials. NEVER commit to Git, and never include in Docker images. Explicitly exclude `.env`, `*.pkl`, and `models/` in `.gitignore`.

## FAQ

### Q: Why is AUC only 0.55-0.60?
A: Stock prediction is inherently difficult. Even large hedge funds typically achieve AUC between 0.55-0.65. Your model is within a reasonable range.

> **Note:** Stock prediction is not a problem with clear answers. Markets are full of noise and randomness. AUC 0.55-0.60 already represents statistically significant predictive ability.

### Q: What does F1 Score mean?
A: F1 = balance between precision and recall. F1 > 0.5 means model is better than random, F1 > 0.6 means usable for trading signals.

### Q: What is Model Ensemble?
A: Trains XGBoost, LightGBM, RandomForest, CatBoost simultaneously, combines their prediction probabilities via VotingClassifier (weighted average), StackingClassifier (meta-model learning), or Blending (out-of-fold stacking). Usually more stable and higher AUC than single models.

### Q: What's the difference between Voting, Stacking, Blending?
A: 
- **Voting**: Combines four models' probabilities via weighted average (default, fastest)
- **Stacking**: Uses a meta-model (LogisticRegression) to learn optimal combination (slower but usually more accurate)
- **Blending**: Similar to Stacking, but uses out-of-fold predictions to avoid overfitting (slowest but usually most accurate)

### Q: What is CatBoost? Why add it?
A: CatBoost is a gradient boosting framework by Yandex, handles categorical features better, usually outperforms XGBoost/LightGBM on financial data. Adding it improves ensemble accuracy.

### Q: What is SMOTE? Why is it needed?
A: SMOTE (Synthetic Minority Over-sampling Technique) generates synthetic samples for minority classes on the training set to solve class imbalance. Applied only on training folds of TimeSeriesSplit — no future data leakage.

> **Warning:** SMOTE must only be applied on training folds, NEVER on validation folds or test sets.

### Q: Can I add more stocks?
A: Edit `STOCK_LIST` in `.env`, e.g. `STOCK_LIST=0700,9988,0005,0939,1810`

### Q: How to view training logs?
A: Logs are at `logs/app.log`

### Q: How long does training take?
A: About 3-5 minutes (parallel training across timeframes). If using Stacking or Blending, about 5-8 minutes.

### Q: How long does prediction take?
A: Parallel prediction across multiple stocks, about 5-10 seconds (depends on stock count).

### Q: What is Feature Correlation Filter?
A: Auto-removes redundant features with |corr| > 0.9 before training. E.g., `ret_3d` is highly correlated with `ret_1d`/`ret_5d` — only the most informative is kept. Reduces noise, speeds up training, lowers overfitting.

### Q: What is Threshold Optimization?
A: Typical systems use fixed thresholds (Buy > 55%, Sell < 45%), but optimal thresholds differ per timeframe. The system auto-searches for Buy/Sell thresholds that maximize F1 on validation, saves them after training.

### Q: What is Model Versioning?
A: Each training saves a timestamped model backup (e.g. `best_model_5d_20260822_163000.pkl`), keeps latest 5. If new model performs poorly, manually rollback.

### Q: How to use Feature Importance CSV?
A: Auto-output after training to `models/feature_importance_{timeframe}.csv`. Open in Excel to analyze which features most influence model predictions, helping feature engineering optimization.

### Q: Does the model auto-update?
A: Requires manual `train_model.py` execution or setting up Windows Task Scheduler

### Q: How to interpret Stop Loss/Take Profit?
A: 
- Buy signal: Stop loss is negative (downside stop), take profit is positive (upside gain)
- Sell signal: Stop loss is positive (upside stop), take profit is negative (downside gain)

### Q: What is Model Drift?
A: Model drift means model prediction capability degrades over time. System auto-detects and prompts retraining.

### Q: What is Confidence Calibration?
A: Ensures confidence scores are reliable. If model is over-confident or under-confident, system suggests adjustments.

### Q: What is the Investment Simulator?
A: Investment simulator mimics copy-trading based on historical signals (Buy/Sell/Hold). HKD 20,000 initial capital per stock, buys on Buy, sells on Sell, calculates actual returns, win rate, max drawdown. Switch to "💰 Investment Simulator" in the Streamlit sidebar.

### Q: Do simulation results include trading costs?
A: Yes. Simulation includes HK standard trading costs: 0.1% commission (min HKD 20) + 0.13% stamp duty (sell-side only). P/L is net of costs.

### Q: Can the simulator short-sell?
A: No. Sell signals only close existing positions — no short selling. If a Sell signal appears with no holdings, no trade is executed.

### Q: What is Sharpe Ratio?
A: Sharpe Ratio is annualized risk-adjusted return. Calculated as (mean return - risk-free rate) / return std × √252. Generally: >1 is good, >2 is great, <0 means worse than risk-free investment.

### Q: What is Sortino Ratio?
A: Similar to Sharpe Ratio, but only considers downside risk (std of negative returns). Suitable for evaluating asymmetric return strategies.

### Q: What is Profit Factor?
A: Profit Factor = total profit / total loss. >1 means profit exceeds loss, >2 is great. If < 1, the strategy is overall losing.

### Q: What is Buy & Hold Benchmark?
A: Buy & Hold is the simplest strategy: buy at start and hold to end. System compares signal strategy return to buy & hold, showing excess return (Alpha). Positive Alpha means strategy outperforms the market.

### Q: What is Portfolio Simulation?
A: Portfolio simulation allocates capital equally across all selected stocks, simulates multi-stock portfolio performance. Shows diversification effect, portfolio Sharpe Ratio and max drawdown.

### Q: What is Confidence-Weighted Strategy?
A: Confidence-weighted strategy adjusts position size by signal confidence. High confidence signals (e.g., 90%) invest more capital (up to 100%), low confidence signals (e.g., 50%) invest less (min 30%). This lets you invest more in higher-conviction trades.

### Q: What is Monte Carlo Test?
A: Monte Carlo test randomly flips signals (e.g., 20% of Buy/Sell signals randomly swapped), runs 1000 simulations. Tests strategy robustness: if strategy remains profitable after random signal disruption, it's more reliable.

### Q: How is Prediction Accuracy Calculated?
A: System validates historical predictions: Buy signal → did N-day closing price rise? Sell signal → did it fall? Accuracy = correct predictions / total predictions × 100%. Uses actual price data, not just confidence as proxy.

### Q: What's the difference between Win Rate and Prediction Accuracy?
A: Win rate is the proportion of profitable trades in the simulation (calculated at sell or expiry). Prediction accuracy is whether the signal direction was correct (price moved as predicted). They can differ because win rate is also affected by trading costs, entry/exit timing, etc.

### Q: What is Model Disagreement?
A: Model disagreement is the degree of inconsistency among the four models (XGBoost, LightGBM, RandomForest, CatBoost). For example:
- `4/0`: All four say Buy → disagreement 0 (full agreement)
- `3/1`: Three Buy, one Sell → disagreement 0.25
- `2/2`: Two Buy, two Sell → disagreement 0.5 (split)
System forces Hold when disagreement >= 50% to avoid weak decisions when models disagree.

### Q: Why force Hold on Model Disagreement?
A: When models disagree (e.g., 2v2), market direction is unclear. Buy/Sell decisions carry higher risk. Forced Hold avoids entering during uncertainty, protecting capital. Dashboard shows ⚠️ warning indicating disagreement level.

### Q: How to read Buy/Sell signals on K-line chart?
A: K-line chart marks model-predicted buy (▲ green) and sell (▼ red) signal positions. Triangle position corresponds to prediction day price — you can visually see if price moved in the predicted direction after the signal. E.g., price rising after Buy means correct prediction.

### Q: What is the role of Technical Indicators?
A: Technical indicators (RSI, MACD, Stochastic, etc.) are model input features. Dashboard shows latest values, helping you understand if technicals support the model's Buy/Sell decision. E.g., a Buy signal when RSI > 70 (overbought) warrants caution.

### Q: What is Signal Confirmation Analysis?
A: Each Buy/Sell signal auto-analyzes whether current indicators support it. For example:
- ✅ RSI 42 low → supports Buy
- ❌ MACD -1.06 bearish momentum → contradicts Buy
- ➖ ADX 9 ranging → signal reliability reduced
Finally shows "X support, X neutral, X contradict" to quickly assess signal reliability. Hold signals do not display analysis.

### Q: Are Technical Indicators calculated in real-time?
A: Yes. Indicators fetch 2 years of OHLCV data from yfinance in real-time, computed via `compute_features()`. Results cached 5 minutes (TTL=300s) to avoid recomputation. If market hasn't closed today, today's Close may be NaN — system auto-skips.

### Q: How is Rolling Accuracy calculated?
A: Rolling accuracy = correct predictions in last 30 days / total predictions × 100%. Verified by: Buy signal → did N-day closing rise? Sell signal → did it fall? Uses actual price data, more reliable than training metrics (F1/AUC) alone.

### Q: How to run tests?
A: Run tests with pytest: `python -m pytest tests/ -v`. Tests cover env config, feature engineering, model training, prediction upload core functionality.

### Q: What features are covered by tests?
A: 87 tests covering:
- Env var loading & validation (10)
- Technical indicator calculation: RSI, MACD, ATR, ADX, Stochastic, MFI, Williams %R (17)
- Model training: XGBoost, LightGBM, RandomForest, CatBoost (incl. early stopping), SMOTE, Blending (14)
- Prediction: date calculation, model loading, signal determination, upload (10)
- Extended features: sentiment, sector, short selling, connect flow, regime, online learning, dynamic weighting (26)
- Dashboard pages: backtest page, portfolio page (4)
- News sentiment real data verification (2)

### Q: What are the 15 Extended Features?
A: Extended features added in two phases, fetching additional data from external sources:

**Phase 1 (10 Features)：**
- **Sentiment (3)**: Market sentiment score from news/social media
- **Sector Rotation (3)**: Tracks sector ETF momentum for fund rotation
- **Short Selling (3)**: Monitors short selling activity — high shorting may signal bearish sentiment
- **Connect Flow (1)**: Southbound capital flow reflects mainland investor sentiment

**Phase 2 (5 Features)：**
- **Regime (3)**: Uses HSI MA50/MA200 crossover for bull/bear/sideways detection

All extended features fall back to defaults when data is unavailable — core functionality is unaffected.

### Q: How does Sentiment Analysis work?
A: Sentiment analysis fetches stock news from East Money (AKShare), calculates sentiment scores using keyword matching. Positive keywords (e.g., "rise", "breakthrough", "good news") add points, negative keywords (e.g., "fall", "breakdown", "bad news") subtract. Score range: -1 (extremely bearish) to +1 (extremely bullish).

**Feature Details:**
| Feature | Description | Purpose |
|---|---|---|
| `sentiment_5d` | 5-day rolling sentiment average | Short-term market sentiment |
| `sentiment_10d` | 10-day rolling sentiment average | Mid-term market sentiment |
| `sentiment_change` | Sentiment change (5d - 10d) | Sentiment momentum, positive = improving |

**Notes:**
- News titles are shorter than content, keyword matching more precise
- Falls back to 0 (neutral) when no news available
- Cached 4 hours to avoid repeated requests

### Q: How does Sector Rotation work?
A: Sector rotation tracks momentum of 6 Hang Seng sector ETFs (Tech, Finance, Property, Energy, Healthcare, Consumer) to identify fund rotation. When a sector's momentum exceeds HSI, funds are flowing into that sector.

**Sector ETF Mapping:**
| Sector | ETF Code | Description |
|---|---|---|
| Tech | `3033.HK` | Hang Seng TECH Index ETF |
| Finance | `3086.HK` | Hang Seng Mainland Banks ETF |
| Property | `3097.HK` | Hang Seng Properties ETF |
| Energy | `3046.HK` | Hang Seng Energy ETF |
| Healthcare | `3069.HK` | Hang Seng Healthcare ETF |
| Consumer | `3053.HK` | Hang Seng Consumer ETF |

**Feature Details:**
| Feature | Description | Purpose |
|---|---|---|
| `sector_momentum_5d` | 5-day sector ETF momentum | Short-term sector rotation |
| `sector_momentum_20d` | 20-day sector ETF momentum | Mid-term sector rotation |
| `sector_vs_hsi` | Sector vs HSI performance | Sector excess return |

### Q: How does Short Selling work?
A: Short selling ratio tracks market-wide shorting activity. High shorting may signal bearish sentiment, but extreme shorting may trigger a short squeeze. Uses yfinance HSI data and volume patterns as proxy.

**Feature Details:**
| Feature | Description | Purpose |
|---|---|---|
| `short_sell_ratio` | Real-time short selling ratio | Current bearish sentiment |
| `short_sell_ratio_5d` | 5-day rolling short sell ratio | Short-term shorting trend |
| `short_sell_ratio_change` | Short sell ratio change | Shorting momentum, positive = increasing |

**Notes:**
- HK short selling data has T+1 delay, uses volume patterns as proxy
- Extreme shorting (>0.8) may trigger short squeeze, actually bullish

### Q: How does Connect Flow work?
A: Connect flow tracks southbound capital (mainland investors buying HK stocks). Increased southbound inflow indicates mainland bullishness on HK stocks. Uses HSI volume and price patterns as proxy.

**Feature Details:**
| Feature | Description | Purpose |
|---|---|---|
| `southbound_net_5d` | 5-day net southbound flow | Short-term capital flow |
| `southbound_momentum` | Southbound flow momentum | Capital inflow acceleration |
| `connect_sentiment` | Connect market sentiment | Mainland investor sentiment |

**Sentiment Score Logic:**
| Condition | Score | Description |
|---|---|---|
| HSI 5-day return > 1% | +0.3 | Moderate HSI rise |
| HSI 5-day return > 3% | +0.5 | Strong HSI rise |
| HSI 5-day return < -1% | -0.3 | Moderate HSI decline |
| HSI 5-day return < -3% | -0.5 | Strong HSI decline |
| Rising + volume > 1.5x avg | +0.6 | Rising on high volume, strong signal |
| Falling + volume > 1.5x avg | -0.6 | Falling on high volume, weak signal |

### Q: How does Online Learning work?
A: Online learning allows incremental model updates using recent data without full retraining. When last full train was >7 days ago, system auto-updates with last 60 days of data, saving time and compute resources.

**How it Works:**
1. **Trigger**: Last full train > 7 days ago
2. **Training Data**: Last 60 days of history
3. **Update Method**: Uses warm-start to load old model weights, continues training
4. **Model Save**: Updated model overwrites current model

**Notes:**
- Currently disabled by default (USE_ONLINE_LEARNING=False)
- Online learning may cause model drift,建议定期 full retrain
- Only updates XGBoost and LightGBM models

### Q: How does Dynamic Weighting work?
A: Dynamic weighting tracks each model's recent performance, uses EMA to adjust ensemble weights. Better-performing models get higher weights, making ensemble predictions more accurate.

**How it Works:**
1. **Weight Init**: New stock/timeframe starts with equal weights (25% each)
2. **EMA Update**: `New weight = α × recent performance + (1-α) × old weight`
3. **Weight Normalization**: Ensures total weights sum to 1
4. **Min Weight**: Each model has 10% floor to avoid complete exclusion

**EMA Decay Factor:**
- α = 0.3 (default): More weight on recent performance
- α = 0.1: More weight on historical performance
- α = 0.5: Balanced recent vs historical

**Notes:**
- Currently disabled by default (USE_DYNAMIC_WEIGHTING=False)
- Weights stored in `models/dynamic_weights.json`
- Weights auto-update after each prediction

### Q: How does Regime Detection affect predictions?
A: Regime has three states: Bull (MA50>MA200), Bear (MA50<MA200), Sideways (mixed). Buy signals are more reliable in Bull markets, Sell signals in Bear. Dashboard displays current state (🟢/🔴/🟡) for reference.

## Recent Updates

### New Features (2026-09-15)

#### Phase 1: Data Features — 10 New Features

| Feature | Module | Description |
|---|---|---|
| `sentiment_5d` | `src/sentiment.py` | 5-day rolling news sentiment score from AKShare (East Money) |
| `sentiment_10d` | `src/sentiment.py` | 10-day rolling news sentiment score |
| `sentiment_change` | `src/sentiment.py` | Change in sentiment (5d vs 10d) |
| `sector_momentum_5d` | `src/sector.py` | 5-day sector ETF momentum (3033.HK, 3086.HK, 3046.HK, 3069.HK, 3053.HK, 3097.HK) |
| `sector_momentum_20d` | `src/sector.py` | 20-day sector ETF momentum |
| `sector_vs_hsi` | `src/sector.py` | Sector performance relative to HSI |
| `short_sell_ratio` | `src/short_selling.py` | Estimated short selling ratio (based on price/volume patterns) |
| `short_sell_ratio_5d` | `src/short_selling.py` | 5-day rolling short sell ratio |
| `short_sell_ratio_change` | `src/short_selling.py` | Change in short sell ratio |
| `southbound_net_5d` | `src/connect_flow.py` | 5-day net southbound flow |
| `southbound_momentum` | `src/connect_flow.py` | Southbound flow momentum |
| `connect_sentiment` | `src/connect_flow.py` | Connect market sentiment |

#### Phase 2: Model Features — 3 New Features

| Feature | Module | Description |
|---|---|---|
| `market_regime` | `src/regime.py` | Market regime: 0=bear, 1=sideways, 2=bull |
| `regime_confidence` | `src/regime.py` | Confidence of regime classification (0-1) |
| `hsi_trend_50_200` | `src/regime.py` | HSI 50-day MA vs 200-day MA ratio |

#### Phase 3: Dashboard — 2 New Pages

| Page | File | Description |
|---|---|---|
| Backtest | `app/pages/2_backtest.py` | Equity curve, drawdown, trade log, vs buy-and-hold benchmark |
| Portfolio | `app/pages/3_portfolio.py` | Holdings summary, signal/confidence distribution, risk exposure |

#### Model Improvements

| Feature | File | Description |
|---|---|---|
| Online Learning | `src/online_learner.py` | Incremental model updates with warm-start XGBoost/LightGBM |
| Dynamic Weighting | `src/dynamic_weighting.py` | EMA-based ensemble weight adjustment by recent performance |
| Market Regime Display | `app/streamlit_app.py` | Regime indicator in signal cards and performance tab |

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

**8 new test files** — 36 new tests, all 87 tests passing.

### Improvements (2026-09-12)

| Improvement | Description |
|---|---|
| **Tabbed Dashboard** | Dashboard restructured into 5 tabs: Signal Overview, K-line & Indicators, Confidence Trend, Prediction Records, Model Performance |

### Improvements (2026-09-11)

| Improvement | Description |
|---|---|
| **K-line Chart** | Interactive K-line chart with model-predicted buy(▲)/sell(▼) signals, supports custom stock and timeframe (30/60/90/180 days) |
| **Technical Indicators** | Technical indicator panel showing RSI, MACD, Stochastic, ADX, MFI, Bollinger Band Width, ATR, Volume Ratio as signal basis |
| **Signal Confirmation** | Auto-analyzes indicator alignment below each Buy/Sell signal, shows "X support/contradict/neutral" with per-indicator assessment |
| **Rolling Accuracy** | 30-day rolling window for true prediction accuracy, verifies price direction after Buy/Sell |
| **Training Metrics** | Training metrics panel showing F1, AUC, champion model per stock and timeframe |
| **Bug Fix: MFI Missing** | Fixed `get_latest_indicators()` missing `mfi` key causing MFI to always display `—` |
| **Bug Fix: MA50 Deviation** | Fixed MA50 deviation display (raw -0.05% → correct -4.85%) due to missing ×100 |
| **Bug Fix: Accuracy Display** | Fixed rolling accuracy showing 5000.0% (`:.1%` format treated percentage as decimal) |
| **Bug Fix: Wrong Key Name** | Fixed `total_predictions`/`correct_predictions` → `total`/`correct` (matches `calculate_accuracy` return value) |
| **NaN Handling** | Fixed yfinance today's Close NaN causing all indicators to show `—`, now `dropna(subset=["Close"])` first |

**No DB changes** — All features read existing `stock_predictions` table or compute in real-time.

#### Bug Fixes (2026-09-15)

| Bug | Module | Root Cause | Fix |
|---|---|---|---|
| **Sentiment: 0 articles** | `src/sentiment.py` | `str(int("0700"))` → `"700"`, AKShare needs 5-digit format | Use `.zfill(5)` → `"00700"` |
| **Sentiment: all scores = 0** | `src/sentiment.py` | Date column `'发布时间'` not detected → picked `'关键词'` | Added `'时间'`/`'時間'` to date column check |
| **Sentiment: all scores = 0** | `src/sentiment.py` | Text column matched title first, too short for keywords | Prefer content (`'新闻内容'`) over title |
| **Short selling: timedelta error** | `src/short_selling.py` | `fetch_stock_short_selling(code, days=30)` — function expects `years` | Changed to `years=1` |
| **Sector ETF: delisted** | `src/sector.py` | `3022.HK` (finance) delisted on Yahoo Finance | Replaced with `3086.HK` (Hang Seng Mainland Banks) |
| **Sector ETF: delisted** | `src/sector.py` | `3048.HK` (property) delisted on Yahoo Finance | Replaced with `3097.HK` (Hang Seng Properties) |

### Modified Files
- `app/streamlit_app.py` — Tabbed dashboard restructure; added K-line, technical indicators, signal confirmation, rolling accuracy, training metrics; fixed MFI, MA50, accuracy display bugs

### Improvements (2026-09-10)

| Improvement | Description |
|---|---|
| **CatBoost Model** | Added CatBoost as 4th model option, usually better on financial data |
| **Blending Ensemble** | Added Blending mode (out-of-fold stacking), usually more accurate than Voting |
| **Parallel Training** | Timeframes (1d, 5d, 20d) trained in parallel, ~3x speedup |
| **Parallel Prediction** | Multi-stock prediction parallel processing, significant speed improvement |
| **Model Comparison** | Training shows F1 comparison across models, clearly marks winner |
| **Disagreement Detection** | Forces Hold when disagreement >= 50%, shows level (e.g., 2/2) |
| **CatBoost Optimization** | Reduced memory usage (depth 4-6, iterations 200-300, early_stopping_rounds=30, thread_count=4) |
| **GPU Support** | Added `USE_GPU` env var for optional GPU acceleration |
| **Daily Batch** | Improved progress display, fixed Windows compatibility |
| **Unit Tests** | 51 tests covering config, feature engineering, training (incl. early stopping), prediction |
| **New Env Vars** | `USE_CATBOOST=True`, `USE_BLENDING=False`, `USE_GPU=False` |

### Modified Files
- `src/train_model.py` — CatBoost, Blending, parallel training, model comparison, GPU detection, memory optimization
- `src/predict_upload.py` — Parallel prediction, model disagreement calculation, feature alignment fix
- `config.py` — Added USE_CATBOOST, USE_BLENDING, USE_GPU env vars
- `run_daily.bat` — Improved progress display, fixed Windows compatibility
- `.env.example` — Added USE_GPU documentation
- `requirements.txt` — Added catboost>=1.2.0, pytest>=8.0.0
- `app/streamlit_app.py` — Displays model disagreement warning
- `migrate_disagreement.sql` — New DB migration script
- `tests/` — New test directory and 4 test files

### Improvements (2026-09-08)

| Improvement | Description |
|---|---|
| **True Accuracy** | model_monitoring.py now uses actual price data for validation, not confidence as proxy |
| **Backtest Rewrite** | Backtester now actually simulates strategy, calculates real returns, win rate, Sharpe |
| **Win Rate Fix** | Unrealized P/L counted toward win rate when holding at simulation end |
| **Risk Metrics** | Added Sharpe, Sortino, Profit Factor, Avg Holding Days |
| **Benchmark** | Added Buy & Hold baseline, shows excess return (Alpha) |
| **Accuracy Dashboard** | True accuracy per stock and timeframe with bar chart |
| **Portfolio Sim** | Multi-stock portfolio, equal allocation, shows performance |
| **Confidence Weighting** | Position sizing by confidence (30%-100%), high confidence = larger |
| **Monte Carlo** | Random signal flips ×1000, shows distribution, probability, risk metrics |
| **yfinance Priority** | yfinance as primary data source, akshare as fallback |

### Modified Files
- `src/simulator.py` — Added risk metrics, benchmark, portfolio, confidence weighting, Monte Carlo
- `src/model_monitoring.py` — Rewrote calculate_accuracy with real price verification, rewrote backtest engine
- `app/pages/1_💰_投資模擬器.py` — Added risk metrics display, benchmark comparison, accuracy analysis, advanced features
- `src/data_fetcher.py` — yfinance priority, akshare fallback

### Improvements (2026-09-04)

| Improvement | Description |
|---|---|
| **Investment Simulator** | New Streamlit simulator page, custom date range, capital, timeframe |
| **Trading Costs** | Simulation includes 0.1% commission + 0.13% stamp duty |
| **Multi-page Nav** | Sidebar auto-shows page nav (Dashboard + Simulator) |

### Modified Files
- `src/simulator.py` — New investment simulation engine
- `app/pages/1_💰_投資模擬器.py` — New interactive simulator page

### Improvements (2026-09-03)

| Improvement | Description |
|---|---|
| **Logging** | Replaced all `print()` with `setup_logger()`, unified log format |
| **Singleton** | Supabase client now singleton, avoids duplicate connections |
| **Win Rate Fix** | `get_win_rate()` now validates actual price changes (was returning fake data) |
| **MFI Vectorization** | MFI from Python loop to NumPy vectorized (~10x speedup) |
| **Env Check** | Auto-checks `.env` on startup, clear error message if missing |
| **Dependency Pinning** | Added version caps (`<2.0.0`) to prevent future incompatibility |
| **Type Hints** | Added type hints for better maintainability |

### Modified Files
- `config.py` — Added `.env` file existence check
- `src/cleanup_old.py` — Logger, type hints, pre-delete count display
- `src/predict_upload.py` — Singleton Supabase client, fixed win rate calc, type hints
- `src/feature_engineering.py` — MFI vectorized computation
- `src/train_model.py` — Added docstrings
- `requirements.txt` — Version pinning, organized by category

## License

This project is for educational and research purposes only. It does not constitute any investment advice. Investing involves risk — invest with caution.

> **Warning:** Past performance does NOT guarantee future results. All backtest and simulation results from this system are for reference only and should NOT be used as the basis for actual investment decisions.
