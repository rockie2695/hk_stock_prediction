"""
Investment simulator - simulates following Buy/Sell signals with a fixed capital.
Tracks portfolio value, trades, and performance metrics.
"""
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Optional
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SUPABASE_URL, SUPABASE_KEY
from src.data_fetcher import fetch_stock_data
from src.logger import setup_logger

logger = setup_logger('simulator')

# Transaction costs (Hong Kong standard)
COMMISSION_RATE = 0.001       # 0.1% commission per trade
MIN_COMMISSION = 20.0         # Minimum HKD 20 per trade
STAMP_DUTY_RATE = 0.0013      # 0.13% stamp duty (sell only)


@dataclass
class Trade:
    date: date
    stock_code: str
    action: str          # 'Buy' or 'Sell'
    price: float
    shares: int
    cost: float          # Transaction cost in HKD
    pnl: float = 0.0     # Realized P&L (only on Sell)


@dataclass
class PortfolioSnapshot:
    date: date
    stock_code: str
    cash: float
    shares: int
    stock_value: float   # shares * current price
    portfolio_value: float  # cash + stock_value


@dataclass
class SimulationResult:
    stock_code: str
    initial_capital: float
    final_value: float
    total_return_pct: float
    total_trades: int
    buy_trades: int
    sell_trades: int
    wins: int
    losses: int
    win_rate: float
    max_drawdown_pct: float
    max_drawdown_date: Optional[date]
    total_commission: float
    total_stamp_duty: float
    trade_log: list = field(default_factory=list)
    portfolio_history: list = field(default_factory=list)


def _calc_commission(amount: float) -> float:
    """Calculate commission fee."""
    return max(amount * COMMISSION_RATE, MIN_COMMISSION)


def _calc_stamp_duty(amount: float) -> float:
    """Calculate stamp duty (sell only)."""
    return amount * STAMP_DUTY_RATE


def _find_optimal_day(
    price_lookup: dict,
    signal_date: date,
    days: int,
    action: str,
) -> tuple:
    """
    Find the optimal day to execute a trade within N days after signal.

    Args:
        price_lookup: Dict mapping date -> close price
        signal_date: Date of the Buy/Sell signal
        days: Number of days to look ahead (5 or 20)
        action: 'Buy' or 'Sell'

    Returns:
        Tuple of (optimal_date, optimal_price)
    """
    all_dates = sorted(price_lookup.keys())

    # Find signal date index
    signal_idx = None
    for i, d in enumerate(all_dates):
        if d >= signal_date:
            signal_idx = i
            break

    if signal_idx is None:
        return signal_date, price_lookup.get(signal_date, 0)

    # Look at next N days (or fewer if near end of data)
    window_end = min(signal_idx + days, len(all_dates))
    window_dates = all_dates[signal_idx:window_end]

    if not window_dates:
        return signal_date, price_lookup.get(signal_date, 0)

    if action == 'Buy':
        # Find the day with the LOWEST close price (best buy price)
        best_date = min(window_dates, key=lambda d: price_lookup[d])
    else:
        # Find the day with the HIGHEST close price (best sell price)
        best_date = max(window_dates, key=lambda d: price_lookup[d])

    return best_date, price_lookup[best_date]


def simulate_investment(
    stock_code: str,
    start_date: str,
    end_date: str,
    initial_capital: float = 20000.0,
    timeframe: str = '1d',
    optimize_timing: bool = False,
) -> Optional[SimulationResult]:
    """
    Simulate investing in a stock by following Buy/Sell signals.

    Args:
        stock_code: Stock code like '0700'
        start_date: Start date string 'YYYY-MM-DD'
        end_date: End date string 'YYYY-MM-DD'
        initial_capital: Starting capital in HKD (default 20000)
        timeframe: Prediction timeframe '1d', '5d', or '20d'
        optimize_timing: If True, find optimal buy/sell day within N-day window
                        (hindsight mode — uses future prices to find best timing)

    Returns:
        SimulationResult or None if no data available
    """
    from supabase import create_client

    client = create_client(SUPABASE_URL, SUPABASE_KEY)

    # Fetch predictions from database
    try:
        result = (
            client.table('stock_predictions')
            .select('prediction_date, signal, confidence, stock_code, timeframe')
            .gte('prediction_date', start_date)
            .lte('prediction_date', end_date)
            .order('prediction_date', desc=False)
            .execute()
        )
        all_predictions = result.data
    except Exception as e:
        logger.error(f"Failed to fetch predictions: {e}")
        return None

    # Filter by stock_code and timeframe in Python (more reliable than Supabase eq filter)
    predictions = [
        p for p in all_predictions
        if str(p.get('stock_code', '')) == str(stock_code) and p.get('timeframe') == timeframe
    ]

    if not predictions:
        logger.warning(f"No predictions found for {stock_code} ({timeframe}) in range {start_date} to {end_date}")
        return None

    # Fetch actual price data
    try:
        price_df = fetch_stock_data(stock_code, years=3)
        price_df['Date'] = pd.to_datetime(price_df['Date']).dt.date
    except Exception as e:
        logger.error(f"Failed to fetch price data for {stock_code}: {e}")
        return None

    # Build lookup: date -> close price
    price_lookup = dict(zip(price_df['Date'], price_df['Close']))

    # Simulate trades
    cash = initial_capital
    shares = 0
    avg_buy_price = 0.0
    trade_log = []
    portfolio_history = []
    total_commission = 0.0
    total_stamp_duty = 0.0
    total_return_pct = 0.0
    wins = 0
    losses = 0

    for pred in predictions:
        pred_date = datetime.strptime(pred['prediction_date'], '%Y-%m-%d').date() if isinstance(pred['prediction_date'], str) else pred['prediction_date']
        signal = pred['signal']

        # Get close price for this date
        close_price = price_lookup.get(pred_date)
        if close_price is None:
            # Try nearest previous date
            prev_dates = [d for d in price_lookup.keys() if d <= pred_date]
            if prev_dates:
                close_price = price_lookup[max(prev_dates)]
            else:
                continue

        # Determine if we should optimize timing
        tf_days = {'1d': 1, '5d': 5, '20d': 20}.get(timeframe, 1)
        use_optimal = optimize_timing and timeframe in ('5d', '20d')

        if signal == 'Buy' and shares == 0:
            if use_optimal:
                # Find optimal buy day within N-day window
                buy_date, buy_price = _find_optimal_day(price_lookup, pred_date, tf_days, 'Buy')
            else:
                buy_date, buy_price = pred_date, close_price

            # Buy: invest all cash
            commission = _calc_commission(cash)
            investable = cash - commission
            shares = int(investable / buy_price)

            if shares > 0:
                cost = shares * buy_price
                total_cost = cost + commission
                cash -= total_cost
                total_commission += commission
                avg_buy_price = buy_price

                trade_log.append(Trade(
                    date=buy_date,
                    stock_code=stock_code,
                    action='Buy',
                    price=buy_price,
                    shares=shares,
                    cost=commission,
                    pnl=0.0,
                ))

                logger.info(f"  BUY  {buy_date} | {shares} shares @ {buy_price:.2f} | Cost: {commission:.2f}")

        elif signal == 'Sell' and shares > 0:
            if use_optimal:
                # Find optimal sell day within N-day window
                sell_date, sell_price = _find_optimal_day(price_lookup, pred_date, tf_days, 'Sell')
            else:
                sell_date, sell_price = pred_date, close_price

            # Sell: close position
            sale_amount = shares * sell_price
            commission = _calc_commission(sale_amount)
            stamp_duty = _calc_stamp_duty(sale_amount)
            net_proceeds = sale_amount - commission - stamp_duty

            pnl = net_proceeds - (shares * avg_buy_price)
            cash += net_proceeds
            total_commission += commission
            total_stamp_duty += stamp_duty

            if pnl > 0:
                wins += 1
            else:
                losses += 1

            trade_log.append(Trade(
                date=sell_date,
                stock_code=stock_code,
                action='Sell',
                price=sell_price,
                shares=shares,
                cost=commission + stamp_duty,
                pnl=pnl,
            ))

            logger.info(f"  SELL {sell_date} | {shares} shares @ {sell_price:.2f} | P&L: {pnl:+.2f}")
            shares = 0
            avg_buy_price = 0.0

        # Record portfolio snapshot at signal date
        stock_value = shares * close_price
        portfolio_value = cash + stock_value

        portfolio_history.append(PortfolioSnapshot(
            date=pred_date,
            stock_code=stock_code,
            cash=cash,
            shares=shares,
            stock_value=stock_value,
            portfolio_value=portfolio_value,
        ))

    # If still holding, use last known price
    if shares > 0:
        last_date = portfolio_history[-1].date if portfolio_history else datetime.strptime(end_date, '%Y-%m-%d').date()
        last_price = price_lookup.get(last_date, avg_buy_price)
        final_value = cash + shares * last_price
    else:
        final_value = cash

    # Calculate metrics
    total_return_pct = ((final_value - initial_capital) / initial_capital) * 100
    total_trades = len(trade_log)
    buy_trades = sum(1 for t in trade_log if t.action == 'Buy')
    sell_trades = sum(1 for t in trade_log if t.action == 'Sell')
    win_rate = (wins / sell_trades * 100) if sell_trades > 0 else 0.0

    # Max drawdown
    max_drawdown_pct = 0.0
    max_drawdown_date = None
    peak = initial_capital
    for snap in portfolio_history:
        if snap.portfolio_value > peak:
            peak = snap.portfolio_value
        drawdown = ((peak - snap.portfolio_value) / peak) * 100
        if drawdown > max_drawdown_pct:
            max_drawdown_pct = drawdown
            max_drawdown_date = snap.date

    result = SimulationResult(
        stock_code=stock_code,
        initial_capital=initial_capital,
        final_value=round(final_value, 2),
        total_return_pct=round(total_return_pct, 2),
        total_trades=total_trades,
        buy_trades=buy_trades,
        sell_trades=sell_trades,
        wins=wins,
        losses=losses,
        win_rate=round(win_rate, 1),
        max_drawdown_pct=round(max_drawdown_pct, 2),
        max_drawdown_date=max_drawdown_date,
        total_commission=round(total_commission, 2),
        total_stamp_duty=round(total_stamp_duty, 2),
        trade_log=trade_log,
        portfolio_history=portfolio_history,
    )

    logger.info(f"  {stock_code} | Final: {final_value:.2f} | Return: {total_return_pct:+.2f}% | Trades: {total_trades} | Win Rate: {win_rate:.1f}%")
    return result


def simulate_all_stocks(
    stock_codes: list,
    start_date: str,
    end_date: str,
    initial_capital: float = 20000.0,
    timeframe: str = '1d',
    optimize_timing: bool = False,
) -> dict:
    """
    Run simulation for multiple stocks.

    Args:
        stock_codes: List of stock codes
        start_date: Start date string 'YYYY-MM-DD'
        end_date: End date string 'YYYY-MM-DD'
        initial_capital: Starting capital per stock in HKD
        timeframe: Prediction timeframe
        optimize_timing: If True, find optimal buy/sell day within N-day window

    Returns:
        Dict mapping stock_code -> SimulationResult
    """
    results = {}
    for code in stock_codes:
        logger.info(f"Simulating {code}...")
        result = simulate_investment(code, start_date, end_date, initial_capital, timeframe, optimize_timing)
        if result is not None:
            results[code] = result
    return results
