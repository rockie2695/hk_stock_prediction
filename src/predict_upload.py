"""
Daily prediction and upload to Supabase.
Loads the trained model, predicts next-day stock movement, and upserts results.
"""
import os
import sys
import pickle
import pandas as pd
from datetime import datetime, timedelta
import pytz
from supabase import create_client

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import STOCK_LIST, SUPABASE_URL, SUPABASE_KEY
from src.data_fetcher import fetch_stock_data
from src.feature_engineering import compute_features, FEATURE_COLUMNS
from src.logger import setup_logger

logger = setup_logger('predict_upload')
HK_TZ = pytz.timezone('Asia/Hong_Kong')

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'best_model.pkl')


def load_model():
    """Load the trained model from disk."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}\n"
            "Please run 'python src/train_model.py' first."
        )
    with open(MODEL_PATH, 'rb') as f:
        model_data = pickle.load(f)
    logger.info(f"Loaded model: {model_data['model_type']}")
    return model_data


def get_next_trading_day(date: datetime) -> datetime:
    """Get the next trading day (skip weekends)."""
    next_day = date + timedelta(days=1)
    # Skip weekend (5=Saturday, 6=Sunday)
    while next_day.weekday() >= 5:
        next_day += timedelta(days=1)
    return next_day


def predict_stock(stock_code: str, model_data: dict) -> dict:
    """
    Predict next-day movement for a single stock.

    Returns:
        dict with keys: stock_code, prediction_date, signal, confidence
    """
    model = model_data['model']
    model_type = model_data['model_type']
    feature_cols = model_data.get('feature_columns', FEATURE_COLUMNS)

    # Fetch latest data
    df = fetch_stock_data(stock_code, years=1)
    df = compute_features(df)

    # Take the last row with valid features
    valid = df.dropna(subset=feature_cols)
    if valid.empty:
        raise ValueError(f"No valid feature data for {stock_code}")

    latest = valid.iloc[-1:]
    X = latest[feature_cols]

    # Predict
    proba = model.predict_proba(X)[0]
    buy_prob = proba[1]  # Probability of class 1 (up)

    # Determine signal
    if buy_prob > 0.55:
        signal = 'Buy'
    elif buy_prob < 0.45:
        signal = 'Sell'
    else:
        signal = 'Hold'

    # Next trading day
    today = datetime.now(HK_TZ).date()
    prediction_date = get_next_trading_day(datetime.combine(today, datetime.min.time()))

    result = {
        'stock_code': stock_code,
        'prediction_date': prediction_date.date().isoformat(),
        'signal': signal,
        'confidence': float(buy_prob),
        'model_version': today.isoformat(),
    }

    emoji = {'Buy': '📈', 'Sell': '📉', 'Hold': '➡️'}
    logger.info(
        f"{emoji[signal]} {stock_code}: {signal} "
        f"(confidence={buy_prob:.2%}, model={model_type})"
    )
    return result


def upload_to_supabase(records: list):
    """Upload prediction records to Supabase via upsert."""
    if not records:
        logger.warning("No records to upload.")
        return

    try:
        client = create_client(SUPABASE_URL, SUPABASE_KEY)
        logger.info(f"Connected to Supabase, uploading {len(records)} records...")
    except Exception as e:
        logger.error(f"Failed to connect to Supabase: {e}")
        return

    success_count = 0
    fail_count = 0
    for record in records:
        try:
            # Only upload safe fields - never upload model or raw data
            upload_data = {
                'stock_code': record['stock_code'],
                'prediction_date': record['prediction_date'],
                'signal': record['signal'],
                'confidence': record['confidence'],
                'model_version': record['model_version'],
            }
            client.table('stock_predictions').upsert(
                upload_data,
                on_conflict='stock_code,prediction_date'
            ).execute()
            success_count += 1
            logger.info(f"  Uploaded: {record['stock_code']} on {record['prediction_date']}")
        except Exception as e:
            fail_count += 1
            logger.error(f"  Failed to upload {record['stock_code']}: {e}")

    logger.info(f"Upload complete: {success_count} success, {fail_count} failed")


def predict_and_upload():
    """Main pipeline: load model, predict for all stocks, upload to Supabase."""
    logger.info("=== Daily Prediction Started ===")

    # Load model
    try:
        model_data = load_model()
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)

    # Predict for each stock
    records = []
    for code in STOCK_LIST:
        try:
            record = predict_stock(code, model_data)
            records.append(record)
        except Exception as e:
            logger.error(f"Prediction failed for {code}: {e}")
            continue

    if not records:
        logger.error("No predictions generated.")
        sys.exit(1)

    # Upload
    upload_to_supabase(records)
    logger.info("=== Daily Prediction Complete ===")


if __name__ == '__main__':
    predict_and_upload()
