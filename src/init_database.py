"""
Database initialization - creates the stock_predictions table in Supabase PostgreSQL.
Idempotent: safe to run multiple times.
"""
import os
import sys
import psycopg2
import psycopg2.errors

# Ensure project root is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SUPABASE_URL, SUPABASE_DB_PASSWORD
from src.logger import setup_logger

logger = setup_logger('init_database')


def extract_project_ref(supabase_url: str) -> str:
    """Extract project ref from Supabase URL. e.g. 'https://abcde.supabase.co' -> 'abcde'"""
    # Remove trailing slash and protocol
    url = supabase_url.rstrip('/')
    if '://' in url:
        url = url.split('://', 1)[1]
    # Remove .supabase.co suffix
    project_ref = url.split('.')[0]
    return project_ref


def init_database():
    """Create stock_predictions table and indexes (idempotent)."""
    conn = None
    cursor = None
    try:
        project_ref = extract_project_ref(SUPABASE_URL)
        db_url = f"postgresql://postgres:{SUPABASE_DB_PASSWORD}@db.{project_ref}.supabase.co:5432/postgres"

        logger.info(f"Connecting to database (project: {project_ref})...")
        conn = psycopg2.connect(db_url)
        cursor = conn.cursor()

        # Create table
        logger.info("Creating stock_predictions table...")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS stock_predictions (
                id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
                stock_code TEXT NOT NULL,
                prediction_date DATE NOT NULL,
                signal TEXT CHECK (signal IN ('Buy', 'Sell', 'Hold')),
                confidence FLOAT8,
                model_version TEXT,
                created_at TIMESTAMPTZ DEFAULT NOW()
            );
        """)

        # Add UNIQUE constraint (handle if already exists)
        try:
            cursor.execute(
                "ALTER TABLE stock_predictions "
                "ADD CONSTRAINT unique_stock_prediction UNIQUE (stock_code, prediction_date);"
            )
            logger.info("Added UNIQUE constraint on (stock_code, prediction_date).")
        except psycopg2.errors.DuplicateObject:
            logger.info("UNIQUE constraint already exists, skipping.")

        # Create index (idempotent)
        try:
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_stock_date "
                "ON stock_predictions (stock_code, prediction_date DESC);"
            )
            logger.info("Created index idx_stock_date.")
        except Exception as e:
            logger.info(f"Index creation skipped: {e}")

        conn.commit()
        logger.info("Database initialization completed successfully.")

    except psycopg2.OperationalError as e:
        logger.error(f"Database connection failed: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Database initialization error: {e}")
        sys.exit(1)
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


if __name__ == '__main__':
    init_database()
