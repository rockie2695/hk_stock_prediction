-- ============================================
-- 升級腳本：為現有資料表添加 timeframe 欄位
-- 在 Supabase Dashboard > SQL Editor 中執行
-- ============================================

-- Add timeframe column (default '1d' for existing rows)
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'stock_predictions' AND column_name = 'timeframe'
    ) THEN
        ALTER TABLE stock_predictions ADD COLUMN timeframe TEXT DEFAULT '1d';
        ALTER TABLE stock_predictions ADD CONSTRAINT stock_predictions_timeframe_check
            CHECK (timeframe IN ('1d', '5d', '20d'));
    END IF;
END $$;

-- Drop old UNIQUE constraint if exists
ALTER TABLE stock_predictions DROP CONSTRAINT IF EXISTS unique_stock_prediction;

-- Add new UNIQUE constraint with timeframe
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'unique_stock_prediction'
    ) THEN
        ALTER TABLE stock_predictions
        ADD CONSTRAINT unique_stock_prediction UNIQUE (stock_code, prediction_date, timeframe);
    END IF;
END $$;
