"""
Investment simulator - simulates following Buy/Sell signals with a fixed capital.
Tracks portfolio value, trades, and performance metrics.
"""
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Optional
import numpy as np
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
RISK_FREE_RATE = 0.02         # 2% annual risk-free rate


@dataclass
class Trade:
    date: date
    stock_code: str
    action: str          # 'Buy', 'Sell', or 'Hold'
    price: float
    shares: int
    cost: float          # Transaction cost in HKD
    pnl: float = 0.0     # Realized P&L (only on Sell/Hold)


@dataclass
class PortfolioSnapshot:
    date: date
    stock_code: str
    cash: float
    shares: int
    stock_value: float   # shares * current price
    portfolio_value: float  # cash + stock_value


@dataclass
class BenchmarkResult:
    stock_code: str
    initial_capital: float
    final_value: float
    total_return_pct: float
    shares_bought: int
    remaining_cash: float


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
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    profit_factor: float = 0.0
    avg_holding_days: float = 0.0
    benchmark: Optional[BenchmarkResult] = None
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


def _calc_sharpe_ratio(portfolio_values: list, risk_free_rate: float = RISK_FREE_RATE) -> float:
    """Calculate annualized Sharpe ratio from portfolio value time series."""
    if len(portfolio_values) < 2:
        return 0.0

    returns = []
    for i in range(1, len(portfolio_values)):
        if portfolio_values[i-1] > 0:
            returns.append((portfolio_values[i] - portfolio_values[i-1]) / portfolio_values[i-1])

    if not returns or np.std(returns) == 0:
        return 0.0

    daily_rf = risk_free_rate / 252
    excess_returns = np.array(returns) - daily_rf
    return float((np.mean(excess_returns) / np.std(excess_returns)) * np.sqrt(252))


def _calc_sortino_ratio(portfolio_values: list, risk_free_rate: float = RISK_FREE_RATE) -> float:
    """Calculate annualized Sortino ratio (downside deviation only)."""
    if len(portfolio_values) < 2:
        return 0.0

    returns = []
    for i in range(1, len(portfolio_values)):
        if portfolio_values[i-1] > 0:
            returns.append((portfolio_values[i] - portfolio_values[i-1]) / portfolio_values[i-1])

    if not returns:
        return 0.0

    daily_rf = risk_free_rate / 252
    excess_returns = np.array(returns) - daily_rf
    downside = excess_returns[excess_returns < 0]

    if len(downside) == 0 or np.std(downside) == 0:
        return 0.0 if np.mean(excess_returns) <= 0 else 999.0

    return float((np.mean(excess_returns) / np.std(downside)) * np.sqrt(252))


def _calc_profit_factor(trade_log: list) -> float:
    """Calculate profit factor: gross profits / gross losses."""
    gross_profit = sum(t.pnl for t in trade_log if t.pnl > 0)
    gross_loss = sum(abs(t.pnl) for t in trade_log if t.pnl < 0)

    if gross_loss == 0:
        return 999.0 if gross_profit > 0 else 0.0
    return gross_profit / gross_loss


def _calc_avg_holding_days(trade_log: list) -> float:
    """Calculate average holding period in days between Buy and Sell/Hold."""
    holding_days = []
    buy_date = None

    for t in trade_log:
        if t.action == 'Buy':
            buy_date = t.date
        elif t.action in ('Sell', 'Hold') and buy_date is not None:
            days = (t.date - buy_date).days
            holding_days.append(days)
            buy_date = None

    return float(np.mean(holding_days)) if holding_days else 0.0


def simulate_buy_and_hold(
    stock_code: str,
    start_date: str,
    end_date: str,
    initial_capital: float = 20000.0,
    price_lookup: dict = None,
) -> Optional[BenchmarkResult]:
    """
    Simulate a buy-and-hold strategy for benchmark comparison.

    Args:
        stock_code: Stock code like '0700'
        start_date: Start date string 'YYYY-MM-DD'
        end_date: End date string 'YYYY-MM-DD'
        initial_capital: Starting capital in HKD
        price_lookup: Optional pre-fetched price lookup dict

    Returns:
        BenchmarkResult or None if no data available
    """
    if price_lookup is None:
        try:
            price_df = fetch_stock_data(stock_code, years=3)
            price_df['Date'] = pd.to_datetime(price_df['Date']).dt.date
            price_lookup = dict(zip(price_df['Date'], price_df['Close']))
        except Exception as e:
            logger.error(f"Failed to fetch price data for benchmark: {e}")
            return None

    all_dates = sorted(price_lookup.keys())
    start_dt = datetime.strptime(start_date, '%Y-%m-%d').date()
    end_dt = datetime.strptime(end_date, '%Y-%m-%d').date()

    # Find first price on or after start_date
    buy_date = None
    for d in all_dates:
        if d >= start_dt:
            buy_date = d
            break

    if buy_date is None:
        return None

    buy_price = price_lookup[buy_date]

    # Find last price on or before end_date
    sell_date = None
    for d in reversed(all_dates):
        if d <= end_dt:
            sell_date = d
            break

    if sell_date is None:
        return None

    sell_price = price_lookup[sell_date]

    # Buy as many whole shares as possible
    commission = max(initial_capital * COMMISSION_RATE, MIN_COMMISSION)
    investable = initial_capital - commission
    shares = int(investable / buy_price)

    if shares == 0:
        return None

    remaining_cash = initial_capital - (shares * buy_price) - commission
    final_value = remaining_cash + (shares * sell_price)
    total_return = ((final_value - initial_capital) / initial_capital) * 100

    return BenchmarkResult(
        stock_code=stock_code,
        initial_capital=initial_capital,
        final_value=round(final_value, 2),
        total_return_pct=round(total_return, 2),
        shares_bought=shares,
        remaining_cash=round(remaining_cash, 2),
    )


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

    # If still holding, use last known price and count unrealized P&L
    if shares > 0:
        last_date = portfolio_history[-1].date if portfolio_history else datetime.strptime(end_date, '%Y-%m-%d').date()
        last_price = price_lookup.get(last_date, avg_buy_price)
        final_value = cash + shares * last_price

        # Count unrealized P&L as win/loss
        unrealized_pnl = (last_price - avg_buy_price) * shares
        if unrealized_pnl > 0:
            wins += 1
        else:
            losses += 1
        trade_log.append(Trade(
            date=last_date,
            stock_code=stock_code,
            action='Hold',
            price=last_price,
            shares=shares,
            cost=0.0,
            pnl=round(unrealized_pnl, 2),
        ))
    else:
        final_value = cash

    # Calculate metrics
    total_return_pct = ((final_value - initial_capital) / initial_capital) * 100
    total_trades = len(trade_log)
    buy_trades = sum(1 for t in trade_log if t.action == 'Buy')
    sell_trades = sum(1 for t in trade_log if t.action == 'Sell')
    total_exits = wins + losses
    win_rate = (wins / total_exits * 100) if total_exits > 0 else 0.0

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

    # Risk metrics
    portfolio_values = [snap.portfolio_value for snap in portfolio_history]
    sharpe = _calc_sharpe_ratio(portfolio_values)
    sortino = _calc_sortino_ratio(portfolio_values)
    pf = _calc_profit_factor(trade_log)
    avg_hold = _calc_avg_holding_days(trade_log)

    # Buy-and-hold benchmark
    benchmark = simulate_buy_and_hold(stock_code, start_date, end_date, initial_capital, price_lookup)

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
        sharpe_ratio=round(sharpe, 2),
        sortino_ratio=round(sortino, 2),
        profit_factor=round(pf, 2),
        avg_holding_days=round(avg_hold, 1),
        benchmark=benchmark,
        trade_log=trade_log,
        portfolio_history=portfolio_history,
    )

    logger.info(f"  {stock_code} | Final: {final_value:.2f} | Return: {total_return_pct:+.2f}% | Trades: {total_trades} | Win Rate: {win_rate:.1f}% | Sharpe: {sharpe:.2f}")
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


def simulate_portfolio(
    stock_codes: list,
    start_date: str,
    end_date: str,
    total_capital: float = 80000.0,
    timeframe: str = '1d',
    optimize_timing: bool = False,
) -> Optional[SimulationResult]:
    """
    Simulate a combined portfolio across multiple stocks.
    Capital is split equally among stocks.

    Args:
        stock_codes: List of stock codes
        start_date: Start date string 'YYYY-MM-DD'
        end_date: End date string 'YYYY-MM-DD'
        total_capital: Total capital in HKD (split equally)
        timeframe: Prediction timeframe
        optimize_timing: If True, find optimal buy/sell day within N-day window

    Returns:
        SimulationResult for the combined portfolio, or None
    """
    if not stock_codes:
        return None

    capital_per_stock = total_capital / len(stock_codes)
    individual_results = simulate_all_stocks(stock_codes, start_date, end_date, capital_per_stock, timeframe, optimize_timing)

    if not individual_results:
        return None

    # Combine portfolio histories by merging on date
    all_dates = set()
    for r in individual_results.values():
        for snap in r.portfolio_history:
            all_dates.add(snap.date)

    all_dates = sorted(all_dates)

    # Build combined portfolio value over time
    combined_history = []
    for d in all_dates:
        total_value = 0
        for code, r in individual_results.items():
            # Find closest snapshot on or before this date
            snap_value = r.initial_capital / len(stock_codes)
            for snap in r.portfolio_history:
                if snap.date <= d:
                    snap_value = snap.portfolio_value
                else:
                    break
            total_value += snap_value

        combined_history.append(PortfolioSnapshot(
            date=d,
            stock_code='PORTFOLIO',
            cash=0,
            shares=0,
            stock_value=total_value,
            portfolio_value=total_value,
        ))

    # Calculate combined metrics
    final_value = sum(r.final_value for r in individual_results.values())
    total_return_pct = ((final_value - total_capital) / total_capital) * 100

    # Combined trade log
    all_trades = []
    for code, r in individual_results.items():
        all_trades.extend(r.trade_log)
    all_trades.sort(key=lambda t: t.date)

    # Combined wins/losses
    wins = sum(r.wins for r in individual_results.values())
    losses = sum(r.losses for r in individual_results.values())
    total_exits = wins + losses
    win_rate = (wins / total_exits * 100) if total_exits > 0 else 0.0

    # Max drawdown on combined portfolio
    max_drawdown_pct = 0.0
    peak = total_capital
    for snap in combined_history:
        if snap.portfolio_value > peak:
            peak = snap.portfolio_value
        dd = ((peak - snap.portfolio_value) / peak) * 100
        if dd > max_drawdown_pct:
            max_drawdown_pct = dd

    # Risk metrics on combined portfolio
    portfolio_values = [snap.portfolio_value for snap in combined_history]
    sharpe = _calc_sharpe_ratio(portfolio_values)
    sortino = _calc_sortino_ratio(portfolio_values)
    pf = _calc_profit_factor(all_trades)
    avg_hold = _calc_avg_holding_days(all_trades)

    # Combined benchmark
    benchmark = simulate_buy_and_hold(stock_codes[0], start_date, end_date, total_capital)

    total_commission = sum(r.total_commission for r in individual_results.values())
    total_stamp_duty = sum(r.total_stamp_duty for r in individual_results.values())

    return SimulationResult(
        stock_code='PORTFOLIO',
        initial_capital=total_capital,
        final_value=round(final_value, 2),
        total_return_pct=round(total_return_pct, 2),
        total_trades=len(all_trades),
        buy_trades=sum(r.buy_trades for r in individual_results.values()),
        sell_trades=sum(r.sell_trades for r in individual_results.values()),
        wins=wins,
        losses=losses,
        win_rate=round(win_rate, 1),
        max_drawdown_pct=round(max_drawdown_pct, 2),
        max_drawdown_date=None,
        total_commission=round(total_commission, 2),
        total_stamp_duty=round(total_stamp_duty, 2),
        sharpe_ratio=round(sharpe, 2),
        sortino_ratio=round(sortino, 2),
        profit_factor=round(pf, 2),
        avg_holding_days=round(avg_hold, 1),
        benchmark=benchmark,
        trade_log=all_trades,
        portfolio_history=combined_history,
    )


def simulate_with_confidence_weighting(
    stock_code: str,
    start_date: str,
    end_date: str,
    initial_capital: float = 20000.0,
    timeframe: str = '1d',
) -> Optional[SimulationResult]:
    """
    Simulate with position sizing proportional to signal confidence.
    High confidence = larger position, low confidence = smaller position.

    Args:
        stock_code: Stock code like '0700'
        start_date: Start date string 'YYYY-MM-DD'
        end_date: End date string 'YYYY-MM-DD'
        initial_capital: Starting capital in HKD
        timeframe: Prediction timeframe

    Returns:
        SimulationResult or None
    """
    from supabase import create_client

    client = create_client(SUPABASE_URL, SUPABASE_KEY)

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

    predictions = [
        p for p in all_predictions
        if str(p.get('stock_code', '')) == str(stock_code) and p.get('timeframe') == timeframe
    ]

    if not predictions:
        return None

    try:
        price_df = fetch_stock_data(stock_code, years=3)
        price_df['Date'] = pd.to_datetime(price_df['Date']).dt.date
    except Exception as e:
        logger.error(f"Failed to fetch price data for {stock_code}: {e}")
        return None

    price_lookup = dict(zip(price_df['Date'], price_df['Close']))

    cash = initial_capital
    shares = 0
    avg_buy_price = 0.0
    trade_log = []
    portfolio_history = []
    total_commission = 0.0
    total_stamp_duty = 0.0
    wins = 0
    losses = 0

    for pred in predictions:
        pred_date = datetime.strptime(pred['prediction_date'], '%Y-%m-%d').date() if isinstance(pred['prediction_date'], str) else pred['prediction_date']
        signal = pred['signal']
        confidence = pred.get('confidence', 0.5)

        close_price = price_lookup.get(pred_date)
        if close_price is None:
            prev_dates = [d for d in price_lookup.keys() if d <= pred_date]
            if prev_dates:
                close_price = price_lookup[max(prev_dates)]
            else:
                continue

        if signal == 'Buy' and shares == 0:
            # Position size proportional to confidence (30%-100% of capital)
            position_pct = 0.3 + (confidence * 0.7)
            investable = cash * position_pct
            commission = _calc_commission(investable)
            investable -= commission
            shares = int(investable / close_price)

            if shares > 0:
                cost = shares * close_price
                total_cost = cost + commission
                cash -= total_cost
                total_commission += commission
                avg_buy_price = close_price

                trade_log.append(Trade(
                    date=pred_date,
                    stock_code=stock_code,
                    action='Buy',
                    price=close_price,
                    shares=shares,
                    cost=commission,
                    pnl=0.0,
                ))

        elif signal == 'Sell' and shares > 0:
            sale_amount = shares * close_price
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
                date=pred_date,
                stock_code=stock_code,
                action='Sell',
                price=close_price,
                shares=shares,
                cost=commission + stamp_duty,
                pnl=pnl,
            ))

            shares = 0
            avg_buy_price = 0.0

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

    # Handle remaining position
    if shares > 0:
        last_date = portfolio_history[-1].date if portfolio_history else datetime.strptime(end_date, '%Y-%m-%d').date()
        last_price = price_lookup.get(last_date, avg_buy_price)
        final_value = cash + shares * last_price
        unrealized_pnl = (last_price - avg_buy_price) * shares
        if unrealized_pnl > 0:
            wins += 1
        else:
            losses += 1
        trade_log.append(Trade(
            date=last_date,
            stock_code=stock_code,
            action='Hold',
            price=last_price,
            shares=shares,
            cost=0.0,
            pnl=round(unrealized_pnl, 2),
        ))
    else:
        final_value = cash

    total_return_pct = ((final_value - initial_capital) / initial_capital) * 100
    total_trades = len(trade_log)
    buy_trades = sum(1 for t in trade_log if t.action == 'Buy')
    sell_trades = sum(1 for t in trade_log if t.action == 'Sell')
    total_exits = wins + losses
    win_rate = (wins / total_exits * 100) if total_exits > 0 else 0.0

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

    portfolio_values = [snap.portfolio_value for snap in portfolio_history]
    sharpe = _calc_sharpe_ratio(portfolio_values)
    sortino = _calc_sortino_ratio(portfolio_values)
    pf = _calc_profit_factor(trade_log)
    avg_hold = _calc_avg_holding_days(trade_log)
    benchmark = simulate_buy_and_hold(stock_code, start_date, end_date, initial_capital, price_lookup)

    return SimulationResult(
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
        sharpe_ratio=round(sharpe, 2),
        sortino_ratio=round(sortino, 2),
        profit_factor=round(pf, 2),
        avg_holding_days=round(avg_hold, 1),
        benchmark=benchmark,
        trade_log=trade_log,
        portfolio_history=portfolio_history,
    )


def monte_carlo_test(
    stock_code: str,
    start_date: str,
    end_date: str,
    initial_capital: float = 20000.0,
    timeframe: str = '1d',
    n_simulations: int = 1000,
    flip_probability: float = 0.2,
) -> dict:
    """
    Monte Carlo stress test: randomly flip signals and run many simulations.
    Shows distribution of outcomes.

    Args:
        stock_code: Stock code like '0700'
        start_date: Start date string 'YYYY-MM-DD'
        end_date: End date string 'YYYY-MM-DD'
        initial_capital: Starting capital in HKD
        timeframe: Prediction timeframe
        n_simulations: Number of Monte Carlo simulations
        flip_probability: Probability of flipping each signal

    Returns:
        Dict with statistics: mean_return, median_return, percentiles, etc.
    """
    import random
    from supabase import create_client

    client = create_client(SUPABASE_URL, SUPABASE_KEY)

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
        return {'error': str(e)}

    predictions = [
        p for p in all_predictions
        if str(p.get('stock_code', '')) == str(stock_code) and p.get('timeframe') == timeframe
    ]

    if not predictions:
        return {'error': '無預測數據'}

    try:
        price_df = fetch_stock_data(stock_code, years=3)
        price_df['Date'] = pd.to_datetime(price_df['Date']).dt.date
    except Exception as e:
        return {'error': f'無法獲取價格數據: {e}'}

    price_lookup = dict(zip(price_df['Date'], price_df['Close']))

    # First run: original signal returns
    original = simulate_investment(stock_code, start_date, end_date, initial_capital, timeframe)
    original_return = original.total_return_pct if original else 0

    # Run Monte Carlo simulations
    returns = []
    final_values = []

    for _ in range(n_simulations):
        # Create flipped predictions
        flipped = []
        for pred in predictions:
            p = dict(pred)
            if random.random() < flip_probability:
                # Flip the signal
                if p['signal'] == 'Buy':
                    p['signal'] = 'Sell'
                elif p['signal'] == 'Sell':
                    p['signal'] = 'Buy'
            flipped.append(p)

        # Simulate with flipped signals
        cash = initial_capital
        shares = 0
        avg_buy_price = 0.0

        for pred in flipped:
            pred_date = datetime.strptime(pred['prediction_date'], '%Y-%m-%d').date() if isinstance(pred['prediction_date'], str) else pred['prediction_date']
            signal = pred['signal']

            close_price = price_lookup.get(pred_date)
            if close_price is None:
                prev_dates = [d for d in price_lookup.keys() if d <= pred_date]
                if prev_dates:
                    close_price = price_lookup[max(prev_dates)]
                else:
                    continue

            if signal == 'Buy' and shares == 0:
                commission = _calc_commission(cash)
                investable = cash - commission
                shares = int(investable / close_price)
                if shares > 0:
                    cost = shares * close_price
                    cash -= cost + commission
                    avg_buy_price = close_price

            elif signal == 'Sell' and shares > 0:
                sale_amount = shares * close_price
                commission = _calc_commission(sale_amount)
                stamp_duty = _calc_stamp_duty(sale_amount)
                cash += sale_amount - commission - stamp_duty
                shares = 0
                avg_buy_price = 0.0

        # Final value
        last_price = close_price if close_price else avg_buy_price
        final_value = cash + shares * last_price
        total_return = ((final_value - initial_capital) / initial_capital) * 100

        returns.append(total_return)
        final_values.append(final_value)

    returns_array = np.array(returns)
    final_values_array = np.array(final_values)

    return {
        'stock_code': stock_code,
        'n_simulations': n_simulations,
        'flip_probability': flip_probability,
        'original_return_pct': round(original_return, 2),
        'mean_return_pct': round(float(np.mean(returns_array)), 2),
        'median_return_pct': round(float(np.median(returns_array)), 2),
        'std_return_pct': round(float(np.std(returns_array)), 2),
        'percentile_5': round(float(np.percentile(returns_array, 5)), 2),
        'percentile_25': round(float(np.percentile(returns_array, 25)), 2),
        'percentile_75': round(float(np.percentile(returns_array, 75)), 2),
        'percentile_95': round(float(np.percentile(returns_array, 95)), 2),
        'min_return_pct': round(float(np.min(returns_array)), 2),
        'max_return_pct': round(float(np.max(returns_array)), 2),
        'prob_profit': round(float(np.mean(returns_array > 0) * 100), 1),
        'mean_final_value': round(float(np.mean(final_values_array)), 2),
        'returns_distribution': returns,
    }
