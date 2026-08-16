"""
Database initialization - ensures stock_predictions table exists in Supabase.
Checks via REST API. If table missing, provides SQL for Dashboard SQL Editor.
"""
import os
import sys
from supabase import create_client

# Ensure project root is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SUPABASE_URL, SUPABASE_KEY
from src.logger import setup_logger

logger = setup_logger('init_database')

# New table SQL (with timeframe and model metrics)
CREATE_TABLE_SQL = """
-- ============================================
-- 港股預測系統 - 資料表建立腳本
-- 在 Supabase Dashboard > SQL Editor 中執行
-- ============================================

CREATE TABLE IF NOT EXISTS stock_predictions (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    stock_code TEXT NOT NULL,
    prediction_date DATE NOT NULL,
    timeframe TEXT CHECK (timeframe IN ('1d', '5d', '20d')) DEFAULT '1d',
    signal TEXT CHECK (signal IN ('Buy', 'Sell', 'Hold')),
    confidence FLOAT8,
    model_version TEXT,
    model_type TEXT,
    f1_score FLOAT8,
    auc_score FLOAT8,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- UNIQUE constraint (stock_code + prediction_date + timeframe)
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'unique_stock_prediction'
    ) THEN
        ALTER TABLE stock_predictions
        ADD CONSTRAINT unique_stock_prediction UNIQUE (stock_code, prediction_date, timeframe);
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
"""

# Migration SQL for existing tables (add timeframe column)
MIGRATE_SQL = """
-- ============================================
-- 升級腳本：為現有資料表添加 timeframe 欄位
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
"""


def check_table_exists(client) -> bool:
    """Check if stock_predictions table exists via REST API."""
    try:
        response = client.table('stock_predictions').select('id').limit(1).execute()
        return True
    except Exception as e:
        error_msg = str(e).lower()
        if 'pgrst205' in error_msg or 'not found' in error_msg or 'does not exist' in error_msg:
            return False
        return False


def check_has_timeframe(client) -> bool:
    """Check if timeframe column exists by querying it."""
    try:
        response = client.table('stock_predictions').select('timeframe').limit(1).execute()
        return True
    except Exception:
        return False


def init_database():
    """Check if table exists, guide user to create it if not."""
    logger.info("=== Database Initialization ===")

    # Connect via REST API
    logger.info("Connecting to Supabase...")
    try:
        client = create_client(SUPABASE_URL, SUPABASE_KEY)
        logger.info("Connected successfully.")
    except Exception as e:
        logger.error(f"Failed to connect: {e}")
        sys.exit(1)

    # Check if table exists
    logger.info("Checking if stock_predictions table exists...")
    if not check_table_exists(client):
        # Table doesn't exist - provide full CREATE SQL
        logger.info("⚠️  Table does not exist.")
        _save_and_print_sql(CREATE_TABLE_SQL, "create_table.sql")
        return

    # Table exists - check if timeframe column exists
    logger.info("Table exists. Checking timeframe column...")
    if check_has_timeframe(client):
        logger.info("✅ Table is up to date (has timeframe column).")
        return

    # Need migration
    logger.info("⚠️  Missing 'timeframe' column. Running migration...")
    _save_and_print_sql(MIGRATE_SQL, "migrate_table.sql")
    logger.info("After running the migration SQL, re-run this script to verify.")


def _save_and_print_sql(sql: str, filename: str):
    """Save SQL to file and print instructions."""
    sql_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), filename)
    with open(sql_file, 'w', encoding='utf-8') as f:
        f.write(sql)
    logger.info(f"SQL saved to: {sql_file}")
    logger.info("")
    logger.info("=" * 60)
    logger.info("📋 請在 Supabase Dashboard 中執行 SQL:")
    logger.info("")
    logger.info("1. 打開 Supabase Dashboard")
    logger.info("2. 點擊左側選單 SQL Editor")
    logger.info("3. 點擊 New query")
    logger.info("4. 複製 SQL 內容並貼上")
    logger.info("5. 點擊 Run 執行")
    logger.info("6. 執行完成後，重新執行此腳本驗證")
    logger.info("=" * 60)


if __name__ == '__main__':
    init_database()
