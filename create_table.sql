
-- ============================================
-- 港股預測系統 - 資料表建立腳本
-- 在 Supabase Dashboard > SQL Editor 中執行
-- ============================================

CREATE TABLE IF NOT EXISTS stock_predictions (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    stock_code TEXT NOT NULL,
    prediction_date DATE NOT NULL,
    signal TEXT CHECK (signal IN ('Buy', 'Sell', 'Hold')),
    confidence FLOAT8,
    model_version TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- UNIQUE constraint
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'unique_stock_prediction'
    ) THEN
        ALTER TABLE stock_predictions
        ADD CONSTRAINT unique_stock_prediction UNIQUE (stock_code, prediction_date);
    END IF;
END $$;

-- Index
CREATE INDEX IF NOT EXISTS idx_stock_date
ON stock_predictions (stock_code, prediction_date DESC);

-- Enable RLS
ALTER TABLE stock_predictions ENABLE ROW LEVEL SECURITY;

-- Public read policy
DO $$ BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_policies WHERE policyname = 'Allow public read' AND tablename = 'stock_predictions') THEN
        CREATE POLICY "Allow public read" ON stock_predictions FOR SELECT USING (true);
    END IF;
END $$;

-- Public insert policy
DO $$ BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_policies WHERE policyname = 'Allow public insert' AND tablename = 'stock_predictions') THEN
        CREATE POLICY "Allow public insert" ON stock_predictions FOR INSERT WITH CHECK (true);
    END IF;
END $$;

-- Public update policy
DO $$ BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_policies WHERE policyname = 'Allow public update' AND tablename = 'stock_predictions') THEN
        CREATE POLICY "Allow public update" ON stock_predictions FOR UPDATE USING (true);
    END IF;
END $$;
