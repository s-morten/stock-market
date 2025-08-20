

import pandas as pd
import numpy as np
from scipy.stats import zscore
import yfinance as yf
import matplotlib.pyplot as plt
import math
from backtester import run_backtest

def get_reit_data(tickers, start_date='2020-01-01', end_date='2025-01-01'):
    """
    Fetches historical price, dividend, and fundamental data for a list of REIT tickers.
    """
    all_data = {}
    for ticker in tickers:
        print(f"Fetching data for {ticker}...")
        stock = yf.Ticker(ticker)
        
        # Get historical market data
        hist_data = stock.history(start=start_date, end=end_date)
        if hist_data.empty:
            print(f"Could not fetch historical data for {ticker}")
            continue
            
        # Get fundamental data
        balance_sheet = stock.balance_sheet
        if balance_sheet.empty or 'Total Assets' not in balance_sheet.index:
            print(f"Could not fetch balance sheet or 'Total Assets' for {ticker}")
            continue

        # Extract relevant data and align it
        # yfinance provides quarterly data, so we'll need to forward-fill it to match daily prices
        fundamentals = pd.DataFrame(index=hist_data.index)
        
        # Use .get() with a default value to handle missing keys
        total_assets = balance_sheet.loc['Total Assets'] if 'Total Assets' in balance_sheet.index else None
        
        # Accumulated depreciation is not directly available, so we approximate or use related fields.
        # A common proxy is to look at 'Property, Plant and Equipment, Net' vs Gross.
        # If not available, this factor will be limited. For now, we'll use a placeholder if it's missing.
        # Let's check for 'Accumulated Depreciation' or similar fields.
        # Note: yfinance data can be inconsistent here.
        # For simplicity, we will assume 'Total Assets' is a good enough proxy for AUM for now.
        # A more complex implementation would be needed for true adjusted AUM.
        
        shares_outstanding = stock.info.get('sharesOutstanding')
        if not shares_outstanding:
            print(f"Could not fetch shares outstanding for {ticker}")
            # Fallback to basic info if quarterly is not available
            shares_outstanding = stock.info.get('sharesOutstanding')
            if not shares_outstanding:
                print(f"Could not fetch shares outstanding for {ticker}")
                continue

        # Create a combined dataframe
        df = hist_data[['Open', 'High', 'Low', 'Close', 'Dividends']].rename(columns={
            'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Dividends': 'dividend'
        })
        df.index = df.index.tz_localize(None)
        
        # Forward-fill quarterly data to daily frequency
        fund_data = balance_sheet.T[['Total Assets']].resample('D').ffill()
        fund_data.rename(columns={'Total Assets': 'total_assets'}, inplace=True)
        df = df.join(fund_data)
        df['shares_outstanding'] = shares_outstanding
        
        # Forward-fill missing data
        df.ffill(inplace=True);
        
        # For this implementation, we'll skip accumulated_depreciation as it's not reliably available.
        # We will adjust the asset valuation metric accordingly.
        df['accumulated_depreciation'] = 0 # Placeholder

        all_data[ticker] = df.dropna()

    return all_data

def calculate_momentum(data, lookback_period=12, skip_months=1):
    """Calculates the 6-12 month momentum, excluding the most recent month."""
    # Using 21 trading days per month
    monthly_returns = data['close'].pct_change(periods=21)
    momentum = monthly_returns.rolling(window=lookback_period - skip_months).mean().shift(skip_months * 21)
    return momentum

def calculate_dividend_yield(data):
    """Calculates the annualized dividend yield."""
    # yfinance provides per-share dividend, we need to annualize it.
    # We sum up the dividends over the past year (252 trading days)
    annual_dividend = data['dividend'].rolling(window=252, min_periods=1).sum()
    dividend_yield = annual_dividend / data['close']
    return dividend_yield

def calculate_asset_valuation(data):
    """Calculates the Price-to-AUM Ratio."""
    market_cap = data['close'] * data['shares_outstanding']
    # Since accumulated_depreciation is not reliably available, we use Total Assets as AUM.
    aum = data['total_assets']
    p_aum_ratio = market_cap / aum
    return p_aum_ratio

def generate_signals(all_data, weights={'momentum': 0.4, 'yield': 0.3, 'p_aum': 0.3}):
    """
    Ranks REITs on Momentum + Yield + Asset Valuation and generates long/short signals.
    """
    all_factors = {}
    for ticker, data in all_data.items():
        if data.empty:
            continue
        momentum = calculate_momentum(data)
        dividend_yield = calculate_dividend_yield(data)
        p_aum_ratio = calculate_asset_valuation(data)

        factors = pd.DataFrame({
            'momentum': momentum,
            'yield': dividend_yield,
            'p_aum': p_aum_ratio
        }).dropna()
        
        if factors.empty:
            continue
            
        all_factors[ticker] = factors

    if not all_factors:
        print("Could not calculate factors for any tickers.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # Combine factors for all tickers into a single DataFrame for cross-sectional ranking
    combined_factors = pd.concat(all_factors, axis=1)

    # Normalize factors using z-scores across the universe of REITs at each time step
    # We need to handle potential all-NaN slices
    def zscore_robust(x):
        if x.notna().sum() < 2: # Cannot compute z-score for less than 2 values
            return np.nan
        return zscore(x, nan_policy='omit')

    normalized_factors = combined_factors.groupby(level=1, axis=1).transform(zscore_robust)

    # Calculate composite score
    normalized_factors.columns = pd.MultiIndex.from_tuples(normalized_factors.columns)
    
    composite_score = (normalized_factors.xs('momentum', level=1, axis=1) * weights['momentum'] +
                       normalized_factors.xs('yield', level=1, axis=1) * weights['yield'] -
                       normalized_factors.xs('p_aum', level=1, axis=1) * weights['p_aum']) # Lower P/AUM is better

    # Rank REITs based on the composite score
    ranked_scores = composite_score.rank(axis=1, ascending=False, method='first')

    # Generate signals: long top quintile, short bottom quintile
    num_tickers = len(all_data)
    long_signals = (ranked_scores <= max(1, round(num_tickers * 0.2))).astype(int)
    short_signals = (ranked_scores >= max(1, round(num_tickers * 0.8))).astype(int)

    return long_signals, short_signals, composite_score

def plot_reit_strategy(all_data, long_signals, short_signals, composite_scores, tickers):
    """Plots the price, composite score, and buy/sell signals for each REIT."""
    num_tickers = len(tickers)
    num_cols = 2
    num_rows = math.ceil(num_tickers / num_cols)
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(14, 5 * num_rows), sharex=True)
    axes = axes.flatten()

    for i, ticker in enumerate(tickers):
        ax1 = axes[i]
        
        if ticker not in all_data or ticker not in composite_scores.columns:
            ax1.set_title(f'{ticker} - Data Not Available')
            continue

        ticker_data = all_data[ticker]
        
        # Plot price
        ax1.plot(ticker_data.index, ticker_data['close'], label=f'{ticker} Price', color='black', alpha=0.8)
        ax1.set_ylabel('Price (USD)', color='black')
        ax1.tick_params(axis='y', labelcolor='black')

        # Plot buy signals
        buy_dates = long_signals[long_signals[ticker] == 1].index
        if not buy_dates.empty:
            buy_prices = ticker_data.loc[buy_dates]['close']
            ax1.plot(buy_dates, buy_prices, '^', markersize=10, color='green', label='Long Signal', linestyle='None')

        # Plot sell signals
        sell_dates = short_signals[short_signals[ticker] == 1].index
        if not sell_dates.empty:
            sell_prices = ticker_data.loc[sell_dates]['close']
            ax1.plot(sell_dates, sell_prices, 'v', markersize=10, color='red', label='Short Signal', linestyle='None')
        
        # Create a second y-axis for the composite score
        ax2 = ax1.twinx()
        ax2.plot(composite_scores.index, composite_scores[ticker], label='Composite Score', color='purple', linestyle='--', alpha=0.7)
        ax2.set_ylabel('Composite Score', color='purple')
        ax2.tick_params(axis='y', labelcolor='purple')
        
        ax1.set_title(f'{ticker} - REIT Strategy Analysis')
        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')
        ax1.grid(True)

    for i in range(num_tickers, len(axes)):
        fig.delaxes(axes[i])

    fig.tight_layout()
    plt.savefig('reit_strategy_plot.png')
    print("\nPlot saved as reit_strategy_plot.png")


def analyze_portfolio_turnover(long_signals, short_signals):
    """
    Analyzes how frequently the portfolio composition changes.
    """
    # Combine long and short signals to represent the total desired portfolio state
    # 1 for long, -1 for short, 0 for neutral
    portfolio_state = long_signals - short_signals

    # Calculate the difference in portfolio state from one day to the next
    # A non-zero value indicates a change in signal for that ticker
    daily_changes = portfolio_state.diff()

    # Count how many tickers change signal each day
    num_changes_per_day = (daily_changes != 0).sum(axis=1)

    # Find the days where at least one change occurred
    days_with_changes = num_changes_per_day[num_changes_per_day > 0]

    # Calculate metrics
    total_days = len(portfolio_state)
    num_days_with_changes = len(days_with_changes)
    avg_changes_on_change_days = days_with_changes.mean()
    
    if total_days > 0:
        change_frequency = (num_days_with_changes / total_days) * 100
    else:
        change_frequency = 0

    print("\n--- Portfolio Turnover Analysis ---")
    print(f"Total trading days analyzed: {total_days}")
    print(f"Number of days with at least one portfolio change: {num_days_with_changes}")
    print(f"Portfolio composition changed on {change_frequency:.2f}% of trading days.")
    if not np.isnan(avg_changes_on_change_days):
        print(f"On days with changes, an average of {avg_changes_on_change_days:.2f} positions changed.")
    print("------------------------------------")

    return days_with_changes


def main():
    """Main function to run the REIT strategy."""
    tickers = ["O", "SPG", "AMT", "PLD", "EQIX", "APLE"]
    
    # 1. Get data
    all_data = get_reit_data(tickers, end_date=pd.to_datetime('today').strftime('%Y-%m-%d'))
    
    if not all_data:
        print("No data fetched. Exiting.")
        return
        
    # 2. Generate signals
    long_signals, short_signals, composite_scores = generate_signals(all_data)
    
    if composite_scores.empty:
        print("Could not generate signals.")
        return

    print("--- Composite Scores (Last 5 days) ---")
    print(composite_scores.tail())
    
    print("\n--- Long Signals (Last 5 days) ---")
    print(long_signals.tail())
    
    print("\n--- Short Signals (Last 5 days) ---")
    print(short_signals.tail())

    # 3. Plotting
    plot_reit_strategy(all_data, long_signals, short_signals, composite_scores, tickers)

    # 4. Backtesting
    # The backtester expects column names in the format '{ticker}_signal'
    buy_signals_df = long_signals.rename(columns={t: f"{t}_signal" for t in tickers})
    sell_signals_df = short_signals.rename(columns={t: f"{t}_signal" for t in tickers})

    # For this strategy, a 'short' signal is a signal to sell a long position.
    # The backtester can be used to evaluate this.
    trade_log = run_backtest(all_data, buy_signals_df, sell_signals_df)

    if not trade_log.empty:
        print("\n--- Trade Log Sample ---")
        print(trade_log.head())
        print("------------------------")
        
    # 5. Analyze Turnover
    analyze_portfolio_turnover(long_signals, short_signals)

if __name__ == "__main__":
    main()

