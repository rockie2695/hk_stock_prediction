"""
Daily prediction and upload to Supabase.
Loads 3 models (1d, 5d, 20d), predicts for all timeframes, upserts results.
Supports ensemble models (Voting/Stacking/Blending) and single models.
Includes parallel prediction for multiple stocks.
"""
import os
import sys
import pickle
from typing import Optional
import pandas as pd
from datetime import datetime, timedelta
import pytz
from supabase import create_client, Client
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import STOCK_LIST, SUPABASE_URL, SUPABASE_KEY
from src.data_fetcher import fetch_stock_data
from src.feature_engineering import compute_features, FEATURE_COLUMNS
from src.logger import setup_logger

logger = setup_logger('predict_upload')
HK_TZ = pytz.timezone('Asia/Hong_Kong')

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
TIMEFRAMES = {'1d': 1, '5d': 5, '20d': 20}

# Module-level Supabase client (created once)
_supabase_client: Optional[Client] = None


def get_supabase_client() -> Client:
    """Get or create a singleton Supabase client."""
    global _supabase_client
    if _supabase_client is None:
        _supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
    return _supabase_client


def load_models() -> dict:
    """
    Load all 3 trained models from disk.
    
    Returns:
        Dict mapping timeframe labels ('1d', '5d', '20d') to model data dicts.
        Each model data dict contains: model, model_type, feature_columns, etc.
    """
    models = {}
    for label in TIMEFRAMES:
        path = os.path.join(MODELS_DIR, f'best_model_{label}.pkl')
        if not os.path.exists(path):
            logger.warning(f"Model not found: {path}")
            continue
        with open(path, 'rb') as f:
            models[label] = pickle.load(f)
        model_type = models[label]['model_type']
        logger.info(f"Loaded model: {label} ({model_type})")
    return models


def get_prediction_date(days_ahead: int) -> str:
    """Get the prediction target date (count only business days, skip weekends)."""
    today = datetime.now(HK_TZ).date()
    target = today
    count = 0
    while count < days_ahead:
        target += timedelta(days=1)
        if target.weekday() < 5:  # Monday=0 to Friday=4
            count += 1
    return target.isoformat()


def get_previous_confidence(client: Client, stock_code: str, timeframe: str) -> Optional[float]:
    """Get the previous confidence for a stock and timeframe from database.
    
    Args:
        client: Supabase client
        stock_code: Stock code like '0700'
        timeframe: '1d', '5d', or '20d'
        
    Returns:
        Previous confidence value or None if not found
    """
    try:
        result = client.table('stock_predictions').select('confidence').eq(
            'stock_code', stock_code
        ).eq(
            'timeframe', timeframe
        ).order('created_at', desc=True).limit(1).execute()
        
        if result.data and len(result.data) > 0:
            return result.data[0]['confidence']
    except Exception as e:
        logger.warning(f"  Could not get previous confidence: {e}")
    
    return None


def get_win_rate(client: Client, stock_code: str, timeframe: str) -> dict:
    """
    Calculate win rate for past predictions by verifying actual price outcomes.
    
    Args:
        client: Supabase client
        stock_code: Stock code like '0700'
        timeframe: '1d', '5d', or '20d'
        
    Returns:
        Dict with win_rate, total, wins, buy_signals, sell_signals
    """
    try:
        # Get past predictions (excluding today) with their target dates
        today = datetime.now(HK_TZ).date().isoformat()
        result = client.table('stock_predictions').select(
            'signal', 'prediction_date', 'timeframe'
        ).eq(
            'stock_code', stock_code
        ).eq(
            'timeframe', timeframe
        ).lt('prediction_date', today).order('prediction_date', desc=True).limit(30).execute()
        
        if not result.data or len(result.data) < 3:
            return {'win_rate': None, 'total': 0, 'wins': 0}
        
        # Fetch actual price data to verify signals
        try:
            df = fetch_stock_data(stock_code, years=1)
            df['Date'] = pd.to_datetime(df['Date']).dt.date
        except Exception as e:
            logger.warning(f"  Could not fetch price data for win rate: {e}")
            return {'win_rate': None, 'total': 0, 'wins': 0}
        
        total = 0
        wins = 0
        buy_count = 0
        sell_count = 0
        
        for pred in result.data:
            signal = pred['signal']
            pred_date = pred['prediction_date']
            tf = pred['timeframe']
            
            if signal == 'Hold':
                continue
            
            total += 1
            if signal == 'Buy':
                buy_count += 1
            else:
                sell_count += 1
            
            # Get the timeframe days
            days = TIMEFRAMES.get(tf, 1)
            
            # Find prediction date and target date in price data
            pred_row = df[df['Date'] == pred_date]
            if pred_row.empty:
                continue
            
            pred_idx = pred_row.index[0]
            target_idx = pred_idx + days
            
            if target_idx >= len(df):
                continue
            
            # Check if prediction was correct
            pred_close = df.loc[pred_idx, 'Close']
            target_close = df.loc[target_idx, 'Close']
            
            if signal == 'Buy' and target_close > pred_close:
                wins += 1
            elif signal == 'Sell' and target_close < pred_close:
                wins += 1
        
        win_rate = (wins / total * 100) if total > 0 else 0
        
        return {
            'win_rate': round(win_rate, 1),
            'total': total,
            'wins': wins,
            'buy_signals': buy_count,
            'sell_signals': sell_count
        }
    except Exception as e:
        logger.warning(f"  Could not calculate win rate: {e}")
        return {'win_rate': None, 'total': 0, 'wins': 0}


def fetch_market_features() -> pd.DataFrame:
    """Fetch HSI index and USD/HKD for prediction context."""
    import yfinance as yf
    from datetime import datetime, timedelta

    end_date = datetime.now(HK_TZ)
    start_date = end_date - timedelta(days=90)  # last 90 days (need 20 days for rolling)

    logger.info("Fetching market context for prediction...")

    data = {}

    # HSI index
    try:
        hsi = yf.download('^HSI', start=start_date, end=end_date, progress=False)
        if not hsi.empty:
            if isinstance(hsi.columns, pd.MultiIndex):
                hsi.columns = hsi.columns.get_level_values(0)
            data['hsi_close'] = hsi['Close']
            logger.info(f"  HSI: {len(hsi)} rows")
    except Exception as e:
        logger.warning(f"  HSI fetch failed: {e}")

    # USD/HKD
    try:
        fx = yf.download('HKD=X', start=start_date, end=end_date, progress=False)
        if not fx.empty:
            if isinstance(fx.columns, pd.MultiIndex):
                fx.columns = fx.columns.get_level_values(0)
            data['usdhkd'] = fx['Close']
            logger.info(f"  USD/HKD: {len(fx)} rows")
    except Exception as e:
        logger.warning(f"  USD/HKD fetch failed: {e}")

    if not data:
        return pd.DataFrame()

    market_df = pd.DataFrame(data).ffill()

    # Normalize timezone
    if market_df.index.tz is not None:
        market_df.index = market_df.index.tz_localize(None).normalize()

    if 'hsi_close' in market_df.columns:
        market_df['hsi_ret_5d'] = market_df['hsi_close'].pct_change(5)
        market_df['hsi_ret_20d'] = market_df['hsi_close'].pct_change(20)
    if 'usdhkd' in market_df.columns:
        market_df['usdhkd_change'] = market_df['usdhkd'].pct_change(5)

    market_df = market_df.drop(columns=['hsi_close', 'usdhkd'], errors='ignore')

    logger.info(f"  Market features: {list(market_df.columns)}")
    return market_df


def predict_stock(stock_code: str, models: dict) -> list:
    """
    Predict all 3 timeframes for a single stock.
    
    Args:
        stock_code: Stock code like '0700'
        models: Dict mapping timeframe labels to model data
        
    Returns:
        List of prediction records
    """
    client = get_supabase_client()
    
    # Fetch stock data
    df = fetch_stock_data(stock_code, years=1)
    df = compute_features(df)

    # Ensure Date is datetime and normalized
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None).dt.normalize()

    # Fetch market context
    market_df = fetch_market_features()
    if not market_df.empty:
        market_df = market_df.reset_index()
        if 'Date' not in market_df.columns:
            market_df = market_df.rename(columns={market_df.columns[0]: 'Date'})
        market_df['Date'] = pd.to_datetime(market_df['Date']).dt.tz_localize(None).dt.normalize()

        df = df.merge(market_df, on='Date', how='left')
        df = df.ffill()
        hsi_count = df['hsi_ret_5d'].notna().sum() if 'hsi_ret_5d' in df.columns else 0
        logger.info(f"  Market context merged. Rows with HSI data: {hsi_count}/{len(df)}")

    # Get model's expected features (per timeframe, since each may differ)
    all_features = set()
    for label, model_data in models.items():
        features = model_data.get('feature_columns', FEATURE_COLUMNS)
        all_features.update(features)

    # Use the last valid row - preserve trained feature order per model
    available_features = [c for c in all_features if c in df.columns]
    valid = df.dropna(subset=available_features)
    if valid.empty:
        raise ValueError(f"No valid feature data for {stock_code}")

    logger.info(f"  Using {len(available_features)} features for prediction")

    # Calculate historical volatility for expected return estimation
    hist_vol = valid['ret_1d'].rolling(20).std().iloc[-1] if 'ret_1d' in valid.columns else 0.02
    avg_daily_return = valid['ret_1d'].rolling(20).mean().iloc[-1] if 'ret_1d' in valid.columns else 0.0

    results = []

    for label, model_data in models.items():
        model = model_data['model']
        days = TIMEFRAMES[label]

        # Per-timeframe feature alignment
        model_features = model_data.get('feature_columns', available_features)
        X_features = [c for c in model_features if c in valid.columns]
        X = valid.iloc[-1:][X_features]

        # Predict with available features
        try:
            proba = model.predict_proba(X)[0]
            buy_prob = proba[1]
        except Exception as e:
            logger.error(f"  Prediction error for {stock_code} {label}: {e}")
            continue

        # Calculate model disagreement
        # For ensemble models, check individual model predictions
        model_disagreement = 0.0
        model_split = "0/4"
        
        if hasattr(model, 'estimators_'):
            # Ensemble model - get individual predictions
            individual_predictions = []
            for estimator in model.estimators_:
                try:
                    ind_proba = estimator.predict_proba(X)[0]
                    ind_buy = ind_proba[1]
                    individual_predictions.append(ind_buy)
                except:
                    continue
            
            if individual_predictions:
                # Count how many predict Buy vs Sell
                buy_count = sum(1 for p in individual_predictions if p > 0.5)
                sell_count = len(individual_predictions) - buy_count
                total_models = len(individual_predictions)
                
                # Disagreement = proportion of minority (0 = unanimous, 0.5 = 2v2 split)
                minority_count = min(buy_count, sell_count)
                model_disagreement = minority_count / total_models if total_models > 0 else 0
                model_split = f"{buy_count}/{sell_count}"
        
        # Use optimized thresholds from training (fallback to defaults)
        thresh_buy = model_data.get('threshold_buy', 0.55)
        thresh_sell = model_data.get('threshold_sell', 0.45)

        # Determine signal
        if buy_prob > thresh_buy:
            signal = 'Buy'
        elif buy_prob < thresh_sell:
            signal = 'Sell'
        else:
            signal = 'Hold'
        
        # Force Hold if models disagree >= 50%
        # This prevents weak signals when models are split
        if model_disagreement >= 0.5 and signal != 'Hold':
            original_signal = signal
            signal = 'Hold'
            logger.info(f"  ⚠️ {stock_code} {label}: Models split {model_split}, forcing Hold (was {original_signal})")

        prediction_date = get_prediction_date(days)

        # Calculate expected return
        # Expected return = (confidence - 0.5) * 2 * volatility * sqrt(days)
        confidence_score = buy_prob - 0.5  # -0.5 to +0.5
        expected_return = confidence_score * 2 * hist_vol * (days ** 0.5) * 100  # as percentage

        # Risk/reward ratio
        # Risk = 1 standard deviation move (potential loss if wrong)
        risk = hist_vol * (days ** 0.5) * 100  # as percentage
        
        # Reward = expected return if prediction is correct
        reward = abs(expected_return)
        
        # Risk/Reward ratio = Reward / Risk
        # Since expected_return = confidence_score * 2 * risk,
        # risk_reward = abs(confidence_score * 2 * risk) / risk = abs(confidence_score * 2)
        # This is always 0-1, so let's use a different formula:
        
        # Better approach: Use historical average return as reward
        # and historical volatility as risk
        avg_return = avg_daily_return * days * 100  # average return over N days
        
        # For Buy: reward = expected gain, risk = potential loss
        # For Sell: reward = expected gain (short), risk = potential loss
        if signal == 'Buy':
            reward = max(expected_return, 0.1)  # minimum 0.1% reward
            risk = max(abs(avg_return), 0.1)  # use historical loss as risk
        elif signal == 'Sell':
            reward = max(abs(expected_return), 0.1)
            risk = max(abs(avg_return), 0.1)
        else:
            reward = 0
            risk = 1
        
        risk_reward = reward / risk if risk > 0 else 0

        # Stop-loss and Take-profit
        # Stop-loss: based on 2x ATR or 2 standard deviations
        stop_loss_pct = hist_vol * (days ** 0.5) * 100 * 2  # 2 standard deviations
        take_profit_pct = abs(expected_return) * 1.5  # 1.5x expected return
        
        if signal == 'Buy':
            stop_loss = -stop_loss_pct  # negative for buy
            take_profit = take_profit_pct  # positive for buy
        elif signal == 'Sell':
            stop_loss = stop_loss_pct  # positive for sell (price goes up = loss)
            take_profit = -take_profit_pct  # negative for sell (price goes down = profit)
        else:
            stop_loss = 0
            take_profit = 0

        # Confidence trend (compare with previous prediction)
        prev_confidence = get_previous_confidence(client, stock_code, label)
        if prev_confidence is not None:
            confidence_change = buy_prob - prev_confidence
            if confidence_change > 0.05:
                confidence_trend = "↑"
            elif confidence_change < -0.05:
                confidence_trend = "↓"
            else:
                confidence_trend = "→"
        else:
            confidence_trend = "-"

        # Win rate (historical accuracy)
        win_rate_data = get_win_rate(client, stock_code, label)
        win_rate = win_rate_data.get('win_rate', 0)

        emoji = {'Buy': '📈', 'Sell': '📉', 'Hold': '➡️'}
        logger.info(f"  {emoji[signal]} {stock_code} {label}: {signal} ({buy_prob:.2%}) | Expected: {expected_return:+.2f}% | Risk/Reward: {risk_reward:.2f} | Trend: {confidence_trend} | Win Rate: {win_rate}%")

        results.append({
            'stock_code': stock_code,
            'prediction_date': prediction_date,
            'timeframe': label,
            'signal': signal,
            'confidence': float(buy_prob),
            'model_version': datetime.now(HK_TZ).strftime('%Y-%m-%d'),
            'model_type': model_data.get('model_type', ''),
            'f1_score': model_data.get('f1_score', 0.0),
            'auc_score': model_data.get('auc_score', 0.0),
            'expected_return': round(expected_return, 2),
            'risk_reward': round(risk_reward, 2),
            'stop_loss': round(stop_loss, 2),
            'take_profit': round(take_profit, 2),
            'confidence_trend': confidence_trend,
            'win_rate': win_rate,
            'threshold_buy': model_data.get('threshold_buy', 0.55),
            'threshold_sell': model_data.get('threshold_sell', 0.45),
            'model_disagreement': round(model_disagreement, 2),
            'model_split': model_split,
        })

    return results


def _predict_stock_worker(args):
    """Worker function for parallel prediction of a single stock."""
    stock_code, models = args
    try:
        return predict_stock(stock_code, models)
    except Exception as e:
        logger.error(f"Prediction failed for {stock_code}: {e}")
        return []


def upload_to_supabase(records: list) -> tuple[int, int]:
    """
    Upload prediction records to Supabase via insert (keep history).
    
    Args:
        records: List of prediction record dicts
        
    Returns:
        Tuple of (success_count, fail_count)
    """
    if not records:
        logger.warning("No records to upload.")
        return 0, 0

    try:
        client = get_supabase_client()
        logger.info(f"Connected to Supabase, uploading {len(records)} records...")
    except Exception as e:
        logger.error(f"Failed to connect to Supabase: {e}")
        return 0, len(records)

    success_count = 0
    fail_count = 0
    for record in records:
        try:
            upload_data = {
                'stock_code': record['stock_code'],
                'prediction_date': record['prediction_date'],
                'timeframe': record['timeframe'],
                'signal': record['signal'],
                'confidence': record['confidence'],
                'model_version': record['model_version'],
                'model_type': record['model_type'],
                'f1_score': record['f1_score'],
                'auc_score': record['auc_score'],
                'expected_return': record.get('expected_return', None),
                'risk_reward': record.get('risk_reward', None),
                'stop_loss': record.get('stop_loss', None),
                'take_profit': record.get('take_profit', None),
                'confidence_trend': record.get('confidence_trend', '-'),
                'win_rate': record.get('win_rate', None),
                'threshold_buy': record.get('threshold_buy', 0.55),
                'threshold_sell': record.get('threshold_sell', 0.45),
                'model_disagreement': record.get('model_disagreement', 0.0),
                'model_split': record.get('model_split', '0/0'),
            }

            # Always insert new record (keep history)
            client.table('stock_predictions').insert(upload_data).execute()
            success_count += 1
            logger.info(f"  Uploaded: {record['stock_code']} {record['timeframe']}")
        except Exception as e:
            fail_count += 1
            logger.error(f"  Failed: {record['stock_code']} {record['timeframe']}: {e}")

    logger.info(f"Upload complete: {success_count} success, {fail_count} failed")
    return success_count, fail_count


def predict_and_upload() -> None:
    """
    Main pipeline: load all models, predict for all stocks and timeframes, upload to Supabase.
    Supports parallel prediction for multiple stocks.
    
    Raises:
        SystemExit: If no models found or no predictions generated
    """
    logger.info("=== Daily Prediction Started ===")

    models = load_models()
    if not models:
        logger.error("No models found. Run train_model.py first.")
        sys.exit(1)

    # Parallel prediction for multiple stocks
    all_records = []
    tasks = [(code, models) for code in STOCK_LIST]
    
    logger.info(f"Predicting {len(STOCK_LIST)} stocks in parallel...")
    
    with ProcessPoolExecutor(max_workers=min(len(STOCK_LIST), os.cpu_count() or 1)) as executor:
        future_to_code = {
            executor.submit(_predict_stock_worker, task): task[0] 
            for task in tasks
        }
        
        for future in as_completed(future_to_code):
            code = future_to_code[future]
            try:
                records = future.result()
                all_records.extend(records)
            except Exception as e:
                logger.error(f"Prediction failed for {code}: {e}")

    if not all_records:
        logger.error("No predictions generated.")
        sys.exit(1)

    success, fail = upload_to_supabase(all_records)
    logger.info(f"=== Daily Prediction Complete (uploaded {success}, failed {fail}) ===")


if __name__ == '__main__':
    predict_and_upload()
