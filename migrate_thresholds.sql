-- Add threshold columns for optimized Buy/Sell thresholds
-- Each timeframe has its own optimized thresholds stored per prediction

ALTER TABLE stock_predictions ADD COLUMN IF NOT EXISTS threshold_buy FLOAT8 DEFAULT 0.55;
ALTER TABLE stock_predictions ADD COLUMN IF NOT EXISTS threshold_sell FLOAT8 DEFAULT 0.45;
