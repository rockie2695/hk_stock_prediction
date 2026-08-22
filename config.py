"""
Global configuration - loads .env and provides project-wide variables.
"""
import os
import sys
from dotenv import load_dotenv
import pytz

# Load .env from project root
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(PROJECT_ROOT, '.env'))

# Timezone
HK_TZ = pytz.timezone('Asia/Hong_Kong')

# Required environment variables
SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_KEY')
STOCK_LIST_RAW = os.getenv('STOCK_LIST', '')

# Parse stock list
STOCK_LIST = [code.strip() for code in STOCK_LIST_RAW.split(',') if code.strip()]

# Model training switches (default: enabled)
USE_ENSEMBLE = os.getenv("USE_ENSEMBLE", "True").lower() in ("true", "1", "t")
USE_STACKING = os.getenv("USE_STACKING", "False").lower() in ("true", "1", "t")
USE_SMOTE = os.getenv("USE_SMOTE", "True").lower() in ("true", "1", "t")

# Validation
_missing = []
if not SUPABASE_URL:
    _missing.append('SUPABASE_URL')
if not SUPABASE_KEY:
    _missing.append('SUPABASE_KEY')
if not STOCK_LIST:
    _missing.append('STOCK_LIST')

if _missing:
    raise ValueError(
        f"Missing required environment variables: {', '.join(_missing)}\n"
        f"Please copy .env.example to .env and fill in the values."
    )
