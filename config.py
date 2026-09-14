"""
Global configuration - loads .env and provides project-wide variables.
全域配置 - 從 .env 載入環境變數，提供專案全域變數。
"""
import os
import sys
from dotenv import load_dotenv
import pytz

# Load .env from project root / 從專案根目錄載入 .env
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
_env_path = os.path.join(PROJECT_ROOT, '.env')

# Check if .env file exists before loading / 載入前檢查 .env 是否存在
if not os.path.exists(_env_path):
    print("ERROR: .env file not found! / 錯誤：找不到 .env 檔案！")
    print(f"Please copy .env.example to .env and fill in the values:")
    print(f"  cp .env.example .env")
    print(f"  # Then edit .env with your Supabase credentials")
    sys.exit(1)

load_dotenv(_env_path)

# Timezone / 時區
HK_TZ = pytz.timezone('Asia/Hong_Kong')

# Required environment variables / 必需的環境變數
SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_KEY')
STOCK_LIST_RAW = os.getenv('STOCK_LIST', '')

# Parse stock list / 解析股票列表
STOCK_LIST = [code.strip() for code in STOCK_LIST_RAW.split(',') if code.strip()]

# Model training switches (default: enabled) / 模型訓練開關 (預設：啟用)
USE_ENSEMBLE = os.getenv("USE_ENSEMBLE", "True").lower() in ("true", "1", "t")
USE_STACKING = os.getenv("USE_STACKING", "False").lower() in ("true", "1", "t")
USE_SMOTE = os.getenv("USE_SMOTE", "True").lower() in ("true", "1", "t")
USE_CATBOOST = os.getenv("USE_CATBOOST", "True").lower() in ("true", "1", "t")
USE_BLENDING = os.getenv("USE_BLENDING", "False").lower() in ("true", "1", "t")
USE_GPU = os.getenv("USE_GPU", "False").lower() in ("true", "1", "t")  # GPU for CatBoost / GPU 用於 CatBoost

# Extended feature switches (Phase 1) / 擴展特徵開關 (第一階段)
USE_SENTIMENT = os.getenv("USE_SENTIMENT", "True").lower() in ("true", "1", "t")  # 情緒分析
USE_SECTOR = os.getenv("USE_SECTOR", "True").lower() in ("true", "1", "t")  # 板塊輪動
USE_SHORT_SELL = os.getenv("USE_SHORT_SELL", "True").lower() in ("true", "1", "t")  # 沽空比率
USE_CONNECT = os.getenv("USE_CONNECT", "True").lower() in ("true", "1", "t")  # 機構資金流

# Extended feature switches (Phase 2) / 擴展特徵開關 (第二階段)
USE_ONLINE_LEARNING = os.getenv("USE_ONLINE_LEARNING", "False").lower() in ("true", "1", "t")  # 增量學習
USE_REGIME = os.getenv("USE_REGIME", "True").lower() in ("true", "1", "t")  # 市場狀態偵測
USE_DYNAMIC_WEIGHTING = os.getenv("USE_DYNAMIC_WEIGHTING", "False").lower() in ("true", "1", "t")  # 動態權重

# Validation / 驗證
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
