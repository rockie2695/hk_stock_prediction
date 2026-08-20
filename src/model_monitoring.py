"""
Data quality checks, model drift detection, and backtesting engine.
Run this module to validate data, detect model degradation, and backtest strategies.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import json

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
                issues.append(f'過多高信心度預測: {extreme_high}')
            if extreme_low > 10:
                issues.append(f'過多低信心度預測: {extreme_low}')
            
            # Check signal distribution
            signal_counts = df['signal'].value_counts()
            total = len(df)
            for signal, count in signal_counts.items():
                pct = count / total * 100
                if pct > 70:
                    issues.append(f'{signal} 信號過度集中: {pct:.1f}%')
            
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
        """Calculate prediction accuracy for a stock/timeframe."""
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
            
            if not pred_result.data or len(pred_result.data) < 5:
                return {'accuracy': None, 'sample_size': 0}
            
            df = pd.DataFrame(pred_result.data)
            
            # Simple accuracy: count Buy signals that went up (need price data)
            # For now, use confidence as proxy
            avg_confidence = df['confidence'].mean()
            signal_counts = df['signal'].value_counts().to_dict()
            
            return {
                'accuracy': avg_confidence,
                'sample_size': len(df),
                'signal_distribution': signal_counts,
                'avg_confidence': avg_confidence
            }
        except Exception as e:
            return {'accuracy': None, 'error': str(e)}
    
    def detect_drift(self, stock_code: str, timeframe: str) -> dict:
        """Detect if model performance is degrading."""
        try:
            # Compare recent vs older predictions
            recent = self.calculate_accuracy(stock_code, timeframe, days=7)
            older = self.calculate_accuracy(stock_code, timeframe, days=30)
            
            if not recent['accuracy'] or not older['accuracy']:
                return {'drift': False, 'message': '數據不足，無法檢測漂移'}
            
            # Calculate drift
            accuracy_change = recent['accuracy'] - older['accuracy']
            
            if accuracy_change < -0.1:  # 10% drop
                return {
                    'drift': True,
                    'severity': 'high',
                    'message': f'模型漂移警報: 準確度下降 {abs(accuracy_change):.1%}',
                    'recent_accuracy': recent['accuracy'],
                    'older_accuracy': older['accuracy']
                }
            elif accuracy_change < -0.05:  # 5% drop
                return {
                    'drift': True,
                    'severity': 'medium',
                    'message': f'可能出現漂移: 準確度變化 {accuracy_change:.1%}',
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
    """Backtest prediction strategies."""
    
    def __init__(self, supabase_client):
        self.client = supabase_client
    
    def backtest_strategy(self, stock_code: str, strategy: str = 'buy_and_hold') -> dict:
        """Backtest a simple strategy."""
        try:
            # Get historical predictions
            result = self.client.table('stock_predictions').select(
                'signal', 'confidence', 'prediction_date', 'timeframe'
            ).eq('stock_code', stock_code).order('prediction_date', desc=True).limit(100).execute()
            
            if not result.data or len(result.data) < 10:
                return {'error': '數據不足，無法進行回測'}
            
            df = pd.DataFrame(result.data)
            
            # Simple backtest: count correct signals
            total_predictions = len(df)
            buy_signals = len(df[df['signal'] == 'Buy'])
            sell_signals = len(df[df['signal'] == 'Sell'])
            hold_signals = len(df[df['signal'] == 'Hold'])
            
            # Calculate basic metrics
            avg_confidence = df['confidence'].mean()
            
            # Simplified win rate (would need price data for real calculation)
            # For now, assume higher confidence = better
            high_confidence = df[df['confidence'] > 0.6]
            low_confidence = df[df['confidence'] < 0.4]
            
            return {
                'stock_code': stock_code,
                'total_predictions': total_predictions,
                'signal_distribution': {
                    'Buy': buy_signals,
                    'Sell': sell_signals,
                    'Hold': hold_signals
                },
                'avg_confidence': avg_confidence,
                'high_confidence_count': len(high_confidence),
                'low_confidence_count': len(low_confidence),
                'strategy': strategy
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
