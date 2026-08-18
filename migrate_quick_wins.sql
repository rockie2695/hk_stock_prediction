-- Migration: Add new columns for Quick Wins features
-- Run this SQL in Supabase Dashboard

-- Add stop-loss and take-profit columns
ALTER TABLE stock_predictions ADD COLUMN IF NOT EXISTS stop_loss FLOAT8;
ALTER TABLE stock_predictions ADD COLUMN IF NOT EXISTS take_profit FLOAT8;

-- Add confidence trend column
ALTER TABLE stock_predictions ADD COLUMN IF NOT EXISTS confidence_trend TEXT DEFAULT '-';

-- Add win rate column
ALTER TABLE stock_predictions ADD COLUMN IF NOT EXISTS win_rate FLOAT8;

-- Add comments for documentation
COMMENT ON COLUMN stock_predictions.stop_loss IS 'Suggested stop-loss percentage (negative for Buy, positive for Sell)';
COMMENT ON COLUMN stock_predictions.take_profit IS 'Suggested take-profit percentage (positive for Buy, negative for Sell)';
COMMENT ON COLUMN stock_predictions.confidence_trend IS 'Confidence trend: ↑ (up), ↓ (down), → (stable), - (first prediction)';
COMMENT ON COLUMN stock_predictions.win_rate IS 'Historical win rate percentage for this stock/timeframe';
