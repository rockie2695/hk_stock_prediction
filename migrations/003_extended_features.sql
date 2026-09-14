-- ============================================
-- Database Migration: Extended Features (Phase 1-3)
-- Date: 2026-09-15
-- ============================================
-- 
-- This migration documents the addition of new features to the system.
-- NO SCHEMA CHANGES REQUIRED - all new features are computed in-memory
-- during training and prediction, and are NOT stored in the database.
--
-- The stock_predictions table already contains all necessary columns.
-- New features are used as model inputs but not persisted.
-- ============================================

-- ============================================
-- New Feature Summary
-- ============================================
-- Phase 1 (Data Features):
--   sentiment_5d        - 5-day rolling news sentiment score
--   sentiment_10d       - 10-day rolling news sentiment score
--   sentiment_change    - Change in sentiment (5d vs 10d)
--   sector_momentum_5d  - 5-day sector ETF momentum
--   sector_momentum_20d - 20-day sector ETF momentum
--   sector_vs_hsi       - Sector performance vs HSI
--   short_sell_ratio    - Estimated short selling ratio
--   short_sell_ratio_5d - 5-day rolling short sell ratio
--   short_sell_ratio_change - Change in short sell ratio
--   southbound_net_5d   - 5-day net southbound flow
--   southbound_momentum - Southbound flow momentum
--   connect_sentiment   - Connect market sentiment
--
-- Phase 2 (Model Features):
--   market_regime       - Market regime (0=bear, 1=sideways, 2=bull)
--   regime_confidence   - Confidence of regime classification (0-1)
--   hsi_trend_50_200    - HSI 50-day MA vs 200-day MA ratio
--
-- Phase 3 (Dashboard):
--   No database changes - new pages read from existing stock_predictions table
--   - app/pages/2_backtest.py  - Strategy backtesting page
--   - app/pages/3_portfolio.py - Portfolio view page
-- ============================================

-- Verify existing table structure
SELECT column_name, data_type 
FROM information_schema.columns 
WHERE table_name = 'stock_predictions'
ORDER BY ordinal_position;

-- Expected columns:
-- stock_code, prediction_date, timeframe, signal, confidence,
-- model_version, model_type, f1_score, auc_score,
-- expected_return, risk_reward, stop_loss, take_profit,
-- confidence_trend, win_rate, threshold_buy, threshold_sell,
-- model_disagreement, model_split, created_at
