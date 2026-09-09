-- ============================================
-- 升級腳本：添加模型分歧指標欄位
-- 在 Supabase Dashboard > SQL Editor 中執行
-- ============================================

-- Add model_disagreement column (0 = unanimous, 0.5 = 2v2 split, 1 = complete disagreement)
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'stock_predictions' AND column_name = 'model_disagreement'
    ) THEN
        ALTER TABLE stock_predictions ADD COLUMN model_disagreement FLOAT8 DEFAULT 0;
    END IF;
END $$;

-- Add model_split column (e.g., '4/0', '3/1', '2/2')
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'stock_predictions' AND column_name = 'model_split'
    ) THEN
        ALTER TABLE stock_predictions ADD COLUMN model_split TEXT DEFAULT '0/0';
    END IF;
END $$;
