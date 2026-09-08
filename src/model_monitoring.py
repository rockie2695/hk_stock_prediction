"""
Data quality checks, model drift detection, and backtesting engine.
Run this module to validate data, detect model degradation, and backtest strategies.
"""
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_fetcher import fetch_stock_data

logger = logging.getLogger(__name__)


class DataQualityChecker:
    """Check data quality and completeness."""
    
    def __init__(self, supabase_client):
        self.client = supabase_client
    
    def check_missing_dates(self, stock_code: str, days: int = 30) -> dict:
        """Check for missing prediction dates."""
        try:
            start_date = (datetime.now() - timedelta(days=days)).date().isoformat()
            result = self.client.table('stock_predictions').select(
                'prediction_date', 'timeframe'
            ).eq(
                'stock_code', stock_code
            ).gte('prediction_date', start_date).execute()
            
            if not result.data:
                return {'status': 'warning', 'message': f'過去 {days} 天內沒有 {stock_code} 的預測記錄'}
            
            # Group by timeframe
            df = pd.DataFrame(result.data)
            tf_counts = df.groupby('timeframe').size()
            
            issues = []
            for tf in ['1d', '5d', '20d']:
                count = tf_counts.get(tf, 0)
                if count < 5:
                    issues.append(f'{tf}: 僅有 {count} 筆預測 (至少需要 5 筆)')
            
            if issues:
                return {'status': 'warning', 'message': '預測數量不足', 'issues': issues}
            return {'status': 'ok', 'message': f'{stock_code} 數據品質正常'}
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def check_confidence_distribution(self, stock_code: str) -> dict:
        """Check if confidence distribution is reasonable."""
        try:
            result = self.client.table('stock_predictions').select(
                'confidence', 'signal'
            ).eq('stock_code', stock_code).order('created_at', desc=True).limit(100).execute()
            
            if not result.data:
                return {'status': 'warning', 'message': '沒有足夠數據進行分析'}
            
            df = pd.DataFrame(result.data)
            
            # Check for extreme confidence values
            extreme_high = (df['confidence'] > 0.9).sum()
            extreme_low = (df['confidence'] < 0.1).sum()
            
            issues = []
            if extreme_high > 10:
                issues.append(f'過多高信心度預測 (>90%): {extreme_high} 筆')
            if extreme_low > 10:
                issues.append(f'過多低信心度預測 (<10%): {extreme_low} 筆')
            
            # Check signal distribution
            signal_counts = df['signal'].value_counts()
            total = len(df)
            for signal, count in signal_counts.items():
                pct = count / total * 100
                if pct > 70:
                    signal_name = {'Buy': '買入', 'Sell': '賣出', 'Hold': '持有'}.get(signal, signal)
                    issues.append(f'{signal_name} 信號過度集中: {pct:.1f}% (正常應 <70%)')
            
            if issues:
                return {'status': 'warning', 'message': '信心度分佈異常', 'issues': issues}
            return {'status': 'ok', 'message': '信心度分佈正常'}
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def run_all_checks(self, stock_codes: list) -> list:
        """Run all quality checks for all stocks."""
        results = []
        for code in stock_codes:
            logger.info(f"Running quality checks for {code}...")
            results.append({
                'stock_code': code,
                'missing_dates': self.check_missing_dates(code),
                'confidence_dist': self.check_confidence_distribution(code)
            })
        return results


class ModelDriftDetector:
    """Detect model performance degradation over time."""
    
    def __init__(self, supabase_client):
        self.client = supabase_client
    
    def calculate_accuracy(self, stock_code: str, timeframe: str, days: int = 30) -> dict:
        """Calculate prediction accuracy by verifying against actual price data."""
        try:
            start_date = (datetime.now() - timedelta(days=days)).date().isoformat()

            # Get predictions
            pred_result = self.client.table('stock_predictions').select(
                'signal', 'confidence', 'prediction_date'
            ).eq(
                'stock_code', stock_code
            ).eq(
                'timeframe', timeframe
            ).gte('prediction_date', start_date).execute()

            if not pred_result.data or len(pred_result.data) < 3:
                return {'accuracy': None, 'sample_size': 0, 'message': f'數據不足: 僅 {len(pred_result.data or [])} 筆預測'}

            # Fetch actual price data
            try:
                price_df = fetch_stock_data(stock_code, years=1)
                price_df['Date'] = pd.to_datetime(price_df['Date']).dt.date
                price_lookup = dict(zip(price_df['Date'], price_df['Close']))
                all_dates = sorted(price_lookup.keys())
            except Exception as e:
                logger.error(f"Failed to fetch price data for {stock_code}: {e}")
                return {'accuracy': None, 'sample_size': 0}

            tf_days = {'1d': 1, '5d': 5, '20d': 20}.get(timeframe, 1)
            correct = 0
            total = 0
            buy_correct = 0
            buy_total = 0
            sell_correct = 0
            sell_total = 0

            for pred in pred_result.data:
                pred_date_str = pred['prediction_date']
                if isinstance(pred_date_str, str):
                    pred_date = datetime.strptime(pred_date_str, '%Y-%m-%d').date()
                else:
                    pred_date = pred_date_str

                signal = pred['signal']
                if signal not in ('Buy', 'Sell'):
                    continue

                # Find pred_date index in price data
                pred_idx = None
                for i, d in enumerate(all_dates):
                    if d >= pred_date:
                        pred_idx = i
                        break

                if pred_idx is None or pred_idx + tf_days >= len(all_dates):
                    continue

                pred_close = price_lookup[all_dates[pred_idx]]
                target_close = price_lookup[all_dates[pred_idx + tf_days]]

                total += 1
                is_correct = False
                if signal == 'Buy' and target_close > pred_close:
                    is_correct = True
                elif signal == 'Sell' and target_close < pred_close:
                    is_correct = True

                if is_correct:
                    correct += 1

                if signal == 'Buy':
                    buy_total += 1
                    if is_correct:
                        buy_correct += 1
                elif signal == 'Sell':
                    sell_total += 1
                    if is_correct:
                        sell_correct += 1

            accuracy = (correct / total * 100) if total > 0 else 0
            signal_counts = {p['signal'] for p in pred_result.data}

            return {
                'accuracy': accuracy,
                'correct': correct,
                'total': total,
                'sample_size': len(pred_result.data),
                'signal_distribution': signal_counts,
                'buy_accuracy': (buy_correct / buy_total * 100) if buy_total > 0 else None,
                'sell_accuracy': (sell_correct / sell_total * 100) if sell_total > 0 else None,
                'buy_signals': buy_total,
                'sell_signals': sell_total,
            }
        except Exception as e:
            return {'accuracy': None, 'error': str(e)}
    
    def detect_drift(self, stock_code: str, timeframe: str) -> dict:
        """Detect if model performance is degrading."""
        try:
            recent = self.calculate_accuracy(stock_code, timeframe, days=14)
            older = self.calculate_accuracy(stock_code, timeframe, days=60)

            if recent.get('accuracy') is None or older.get('accuracy') is None:
                msg_recent = recent.get('message', f"樣本={recent.get('sample_size', 0)}")
                msg_older = older.get('message', f"樣本={older.get('sample_size', 0)}")
                return {'drift': False, 'message': f'數據不足，無法檢測漂移 (近期: {msg_recent}, 長期: {msg_older})'}

            accuracy_change = recent['accuracy'] - older['accuracy']

            if accuracy_change < -10:
                return {
                    'drift': True,
                    'severity': 'high',
                    'message': f'模型漂移警報: 準確度下降 {abs(accuracy_change):.1f}%',
                    'recent_accuracy': recent['accuracy'],
                    'older_accuracy': older['accuracy']
                }
            elif accuracy_change < -5:
                return {
                    'drift': True,
                    'severity': 'medium',
                    'message': f'可能出現漂移: 準確度變化 {accuracy_change:.1f}%',
                    'recent_accuracy': recent['accuracy'],
                    'older_accuracy': older['accuracy']
                }
            else:
                return {
                    'drift': False,
                    'message': '模型性能穩定',
                    'recent_accuracy': recent['accuracy'],
                    'older_accuracy': older['accuracy']
                }
        except Exception as e:
            return {'drift': False, 'error': str(e)}
    
    def check_all_models(self, stock_codes: list, timeframes: list = ['1d', '5d', '20d']) -> list:
        """Check drift for all stock/timeframe combinations."""
        results = []
        for code in stock_codes:
            for tf in timeframes:
                drift_result = self.detect_drift(code, tf)
                results.append({
                    'stock_code': code,
                    'timeframe': tf,
                    **drift_result
                })
        return results


class Backtester:
    """Backtest prediction strategies against actual price data."""

    def __init__(self, supabase_client):
        self.client = supabase_client

    def backtest_strategy(self, stock_code: str, strategy: str = 'signal_follow') -> dict:
        """
        Backtest a strategy using actual predictions and price data.

        Args:
            stock_code: Stock code like '0700'
            strategy: 'signal_follow' (buy on Buy, sell on Sell) or 'buy_and_hold'

        Returns:
            Dict with backtest results including returns, win rate, Sharpe ratio
        """
        try:
            # Get historical predictions
            result = self.client.table('stock_predictions').select(
                'signal', 'confidence', 'prediction_date', 'timeframe'
            ).eq('stock_code', stock_code).order('prediction_date', desc=False).limit(200).execute()

            if not result.data or len(result.data) < 10:
                return {'error': '數據不足，無法進行回測'}

            # Fetch actual price data
            try:
                price_df = fetch_stock_data(stock_code, years=2)
                price_df['Date'] = pd.to_datetime(price_df['Date']).dt.date
                price_lookup = dict(zip(price_df['Date'], price_df['Close']))
                all_dates = sorted(price_lookup.keys())
            except Exception as e:
                return {'error': f'無法獲取價格數據: {e}'}

            # Use timeframe=1d for backtesting (most granular)
            predictions = [p for p in result.data if p.get('timeframe') == '1d']
            if not predictions:
                predictions = result.data[:50]

            # Simulate signal-following strategy
            initial_capital = 20000.0
            cash = initial_capital
            shares = 0
            buy_price = 0.0
            trades = []
            portfolio_values = []

            for pred in predictions:
                pred_date_str = pred['prediction_date']
                if isinstance(pred_date_str, str):
                    pred_date = datetime.strptime(pred_date_str, '%Y-%m-%d').date()
                else:
                    pred_date = pred_date_str

                signal = pred['signal']
                close_price = price_lookup.get(pred_date)
                if close_price is None:
                    prev_dates = [d for d in all_dates if d <= pred_date]
                    if prev_dates:
                        close_price = price_lookup[max(prev_dates)]
                    else:
                        continue

                if signal == 'Buy' and shares == 0:
                    shares = int(cash / close_price)
                    if shares > 0:
                        cost = shares * close_price
                        cash -= cost
                        buy_price = close_price
                        trades.append({'date': pred_date, 'action': 'Buy', 'price': close_price, 'shares': shares})

                elif signal == 'Sell' and shares > 0:
                    sale_amount = shares * close_price
                    pnl = sale_amount - (shares * buy_price)
                    cash += sale_amount
                    trades.append({'date': pred_date, 'action': 'Sell', 'price': close_price, 'shares': shares, 'pnl': pnl})
                    shares = 0
                    buy_price = 0.0

                portfolio_values.append(cash + shares * close_price)

            # Final value
            final_price = price_lookup[all_dates[-1]] if all_dates else buy_price
            final_value = cash + shares * final_price

            # Calculate metrics
            total_return = ((final_value - initial_capital) / initial_capital) * 100
            sell_trades = [t for t in trades if t['action'] == 'Sell']
            wins = sum(1 for t in sell_trades if t.get('pnl', 0) > 0)
            win_rate = (wins / len(sell_trades) * 100) if sell_trades else 0

            # Buy and hold benchmark
            first_price = price_lookup[all_dates[0]] if all_dates else 0
            bh_shares = int(initial_capital / first_price) if first_price > 0 else 0
            bh_remaining = initial_capital - (bh_shares * first_price)
            bh_final = bh_remaining + bh_shares * final_price
            bh_return = ((bh_final - initial_capital) / initial_capital) * 100

            # Daily returns for Sharpe
            daily_returns = []
            for i in range(1, len(portfolio_values)):
                if portfolio_values[i-1] > 0:
                    daily_returns.append((portfolio_values[i] - portfolio_values[i-1]) / portfolio_values[i-1])

            sharpe = self.calculate_sharpe_ratio(daily_returns) if daily_returns else 0

            # Max drawdown
            peak = initial_capital
            max_dd = 0
            for v in portfolio_values:
                if v > peak:
                    peak = v
                dd = (peak - v) / peak * 100 if peak > 0 else 0
                if dd > max_dd:
                    max_dd = dd

            return {
                'stock_code': stock_code,
                'strategy': strategy,
                'total_predictions': len(predictions),
                'signal_distribution': {
                    'Buy': len([p for p in predictions if p['signal'] == 'Buy']),
                    'Sell': len([p for p in predictions if p['signal'] == 'Sell']),
                    'Hold': len([p for p in predictions if p['signal'] == 'Hold']),
                },
                'initial_capital': initial_capital,
                'final_value': round(final_value, 2),
                'total_return_pct': round(total_return, 2),
                'total_trades': len(trades),
                'win_rate': round(win_rate, 1),
                'max_drawdown_pct': round(max_dd, 2),
                'sharpe_ratio': round(sharpe, 2),
                'benchmark_return_pct': round(bh_return, 2),
                'benchmark_final_value': round(bh_final, 2),
                'alpha': round(total_return - bh_return, 2),
            }
        except Exception as e:
            return {'error': str(e)}
    
    def calculate_sharpe_ratio(self, returns: list, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio from returns."""
        if not returns or len(returns) < 2:
            return 0
        
        returns_array = np.array(returns)
        avg_return = np.mean(returns_array)
        std_return = np.std(returns_array)
        
        if std_return == 0:
            return 0
        
        sharpe = (avg_return - risk_free_rate) / std_return
        return sharpe


class AlertManager:
    """Manage signal alerts."""
    
    def __init__(self, supabase_client):
        self.client = supabase_client
    
    def check_alerts(self, stock_codes: list) -> list:
        """Check for strong signals that warrant alerts."""
        alerts = []
        
        for code in stock_codes:
            try:
                # Get latest prediction
                result = self.client.table('stock_predictions').select(
                    'signal', 'confidence', 'timeframe', 'expected_return'
                ).eq('stock_code', code).order('created_at', desc=True).limit(3).execute()
                
                if not result.data:
                    continue
                
                for pred in result.data:
                    # Alert conditions
                    if pred['confidence'] > 0.7 and pred['signal'] in ['Buy', 'Sell']:
                        alerts.append({
                            'stock_code': code,
                            'timeframe': pred['timeframe'],
                            'signal': pred['signal'],
                            'confidence': pred['confidence'],
                            'expected_return': pred.get('expected_return', 0),
                            'alert_type': 'strong_signal',
                            'message': f"{code} ({pred['timeframe']}) 出現強勢 {pred['signal']} 信號，信心度 {pred['confidence']:.1%}"
                        })
                    
                    if pred.get('expected_return', 0) and abs(pred['expected_return']) > 5:
                        alerts.append({
                            'stock_code': code,
                            'timeframe': pred['timeframe'],
                            'signal': pred['signal'],
                            'confidence': pred['confidence'],
                            'expected_return': pred['expected_return'],
                            'alert_type': 'high_return',
                            'message': f"{code} ({pred['timeframe']}) 預期報酬 {pred['expected_return']:+.1f}%"
                        })
            except Exception as e:
                logger.error(f"Error checking alerts for {code}: {e}")
        
        return alerts
    
    def format_alerts(self, alerts: list) -> str:
        """Format alerts for display."""
        if not alerts:
            return "目前沒有需要關注的信號。"
        
        lines = ["## 🔔 信號警報\n"]
        for alert in alerts:
            emoji = "📈" if alert['signal'] == 'Buy' else "📉"
            lines.append(f"{emoji} **{alert['stock_code']}** ({alert['timeframe']}): {alert['signal']} - {alert['message']}")
        
        return "\n".join(lines)


class ConfidenceCalibrator:
    """Calibrate confidence scores to be more reliable."""
    
    def __init__(self, supabase_client):
        self.client = supabase_client
    
    def calculate_calibration(self, stock_code: str) -> dict:
        """Calculate calibration metrics."""
        try:
            # Get predictions with outcomes
            result = self.client.table('stock_predictions').select(
                'confidence', 'signal', 'prediction_date'
            ).eq('stock_code', stock_code).order('created_at', desc=True).limit(200).execute()
            
            if not result.data or len(result.data) < 20:
                return {'calibration_score': None, 'message': '數據不足，無法計算校準指標'}
            
            df = pd.DataFrame(result.data)
            
            # Group by confidence buckets
            df['confidence_bucket'] = pd.cut(df['confidence'], bins=10)
            
            # Calculate average confidence per bucket
            calibration = df.groupby('confidence_bucket', observed=True)['confidence'].mean()
            
            # Ideal calibration: confidence should match actual win rate
            # For now, return basic stats
            return {
                'avg_confidence': df['confidence'].mean(),
                'std_confidence': df['confidence'].std(),
                'confidence_range': {
                    'min': df['confidence'].min(),
                    'max': df['confidence'].max()
                },
                'sample_size': len(df)
            }
        except Exception as e:
            return {'calibration_score': None, 'error': str(e)}
    
    def suggest_calibration_adjustment(self, calibration_data: dict) -> dict:
        """Suggest calibration adjustments."""
        if not calibration_data.get('avg_confidence'):
            return {'adjustment': 0, 'message': '無需調整'}
        
        avg = calibration_data['avg_confidence']
        
        # If average confidence is too high or too low, suggest adjustment
        if avg > 0.6:
            return {
                'adjustment': -0.05,
                'message': '模型可能過度自信，建議降低信心度 5%'
            }
        elif avg < 0.4:
            return {
                'adjustment': 0.05,
                'message': '模型可能信心不足，建議提高信心度 5%'
            }
        else:
            return {
                'adjustment': 0,
                'message': '信心度校準正常'
            }
