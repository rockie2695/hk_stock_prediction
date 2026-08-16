-- ============================================
-- 升級腳本：添加 model_type, f1_score, auc_score 欄位
-- 在 Supabase Dashboard > SQL Editor 中執行
-- ============================================

-- Add model_type column
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'stock_predictions' AND column_name = 'model_type'
    ) THEN
        ALTER TABLE stock_predictions ADD COLUMN model_type TEXT;
    END IF;
END $$;

-- Add f1_score column
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'stock_predictions' AND column_name = 'f1_score'
    ) THEN
        ALTER TABLE stock_predictions ADD COLUMN f1_score FLOAT8;
    END IF;
END $$;

-- Add auc_score column
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'stock_predictions' AND column_name = 'auc_score'
    ) THEN
        ALTER TABLE stock_predictions ADD COLUMN auc_score FLOAT8;
    END IF;
END $$;
