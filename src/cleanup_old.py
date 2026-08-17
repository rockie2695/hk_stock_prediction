"""
Cleanup old prediction records (keep last 60 days).
Run daily via run_daily.bat before predictions.
"""
from datetime import datetime, timedelta
from supabase import create_client
from dotenv import load_dotenv
import os

load_dotenv()

def cleanup_old_records(days_to_keep=60):
    """Delete records older than N days."""
    client = create_client(os.getenv('SUPABASE_URL'), os.getenv('SUPABASE_KEY'))
    
    cutoff_date = (datetime.now() - timedelta(days=days_to_keep)).date().isoformat()
    
    try:
        result = client.table('stock_predictions').delete().lt(
            'prediction_date', cutoff_date
        ).execute()
        print(f"Cleaned up records older than {cutoff_date}")
    except Exception as e:
        print(f"Cleanup failed: {e}")

if __name__ == '__main__':
    cleanup_old_records()
