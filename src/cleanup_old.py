"""
Cleanup old prediction records (keep last 60 days).
Run daily via run_daily.bat before predictions.
"""
import os
import sys
from datetime import datetime, timedelta
from supabase import create_client

# Ensure project root is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SUPABASE_URL, SUPABASE_KEY
from src.logger import setup_logger

logger = setup_logger('cleanup_old')


def cleanup_old_records(days_to_keep: int = 60) -> int:
    """
    Delete records older than N days.
    
    Args:
        days_to_keep: Number of days to keep (default 60)
        
    Returns:
        Number of days kept as cutoff
    """
    logger.info(f"Connecting to Supabase...")
    try:
        client = create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception as e:
        logger.error(f"Failed to connect to Supabase: {e}")
        return 0
    
    cutoff_date = (datetime.now() - timedelta(days=days_to_keep)).date().isoformat()
    
    try:
        # First count how many records will be deleted
        count_result = client.table('stock_predictions').select(
            'id', count='exact'
        ).lt('prediction_date', cutoff_date).execute()
        
        delete_count = count_result.count if hasattr(count_result, 'count') else 0
        
        if delete_count == 0:
            logger.info(f"No records older than {cutoff_date} to clean up.")
            return days_to_keep
        
        # Delete old records
        result = client.table('stock_predictions').delete().lt(
            'prediction_date', cutoff_date
        ).execute()
        logger.info(f"Cleaned up {delete_count} records older than {cutoff_date}")
        return days_to_keep
    except Exception as e:
        logger.error(f"Cleanup failed: {e}")
        return 0

if __name__ == '__main__':
    logger.info("=== Cleanup Started ===")
    days = cleanup_old_records()
    logger.info(f"=== Cleanup Complete (kept {days} days) ===")
