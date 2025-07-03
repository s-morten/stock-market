import pandas as pd
from binance.client import Client
import os
import numpy as np
import matplotlib.pyplot as plt
import math

# --- Configuration ---
# It's recommended to use environment variables for API keys
# api_key = os.environ.get('BINANCE_API_KEY')
# api_secret = os.environ.get('BINANCE_API_SECRET')

# Or, uncomment and hardcode them (not recommended for production)
# api_key = "YOUR_API_KEY"
# api_secret = "YOUR_API_SECRET"

client = Client()

# --- Data Fetching and Caching (Reused from trading_system.py) ---

def get_historical_data(symbol, interval, lookback):
    """Fetches historical kline data from Binance, including all necessary columns."""
    try:
        print(f"Fetching fresh data for {symbol}...")
        klines = client.get_historical_klines(symbol, interval, lookback)
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
            
        return df[['open', 'high', 'low', 'close', 'volume']]
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        return pd.DataFrame()

def load_or_fetch_data(tickers, interval, lookback, cache_filename="crypto_data_cache.csv"):
    """Loads data from a CSV cache if it's valid, otherwise fetches from Binance."""
    data_cols = ['open', 'high', 'low', 'close', 'volume']
    required_cache_cols = [f"{t}_{c}" for t in tickers for c in data_cols]

    try:
        if os.path.exists(cache_filename):
            print(f"Cache file found: {cache_filename}")
            cached_df = pd.read_csv(cache_filename, index_col='timestamp', parse_dates=True)
            
            missing_cols = [c for c in required_cache_cols if c not in cached_df.columns]
            
            is_stale = False
            if not cached_df.empty:
                last_cached_date = cached_df.index.max().date()
                yesterday = (pd.Timestamp.utcnow() - pd.Timedelta(days=1)).date()
                if last_cached_date < yesterday:
                    is_stale = True
                    print("Cache is stale.")

            if not missing_cols and not is_stale:
                print("Cache is valid. Loading data from cache.")
                all_data = {}
                for ticker in tickers:
                    ticker_cols = {f"{ticker}_{c}": c for c in data_cols}
                    all_data[ticker] = cached_df[ticker_cols.keys()].rename(columns=ticker_cols)
                return all_data
            else:
                if missing_cols:
                    print(f"Columns missing from cache: {missing_cols[:5]}...")
                print("Cache is invalid. Fetching all data from API.")
        else:
            print("No cache file found.")
    except Exception as e:
        print(f"Error reading cache file: {e}. Fetching fresh data.")

    all_data_fetch = {}
    for ticker in tickers:
        data = get_historical_data(ticker, interval, lookback)
        if not data.empty:
            all_data_fetch[ticker] = data

    if not all_data_fetch:
        return {}

    combined_for_cache = pd.DataFrame()
    for ticker, df in all_data_fetch.items():
        for col in df.columns:
            combined_for_cache[f"{ticker}_{col}"] = df[col]
            
    combined_for_cache.to_csv(cache_filename)
    print(f"Data fetched and saved to {cache_filename}")
    
    return all_data_fetch

# --- Plotting (Reused and adapted from trading_system.py) ---

def plot_data_with_smas(data, buy_signals_df, sell_signals_df, tickers, fast_sma, slow_sma):
    """Plots the price data, SMAs, and both buy and sell signals for multiple tickers."""
    num_tickers = len(tickers)
    num_cols = 2
    num_rows = math.ceil(num_tickers / num_cols)
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(14, 5 * num_rows), sharex=True)
    axes = axes.flatten()

    for i, ticker in enumerate(tickers):
        ax = axes[i]
        ticker_data = data[ticker]
        
        # Plot price and SMAs
        ax.plot(ticker_data.index, ticker_data['close'], label=f'{ticker} Price', color='black', alpha=0.8)
        ax.plot(ticker_data.index, ticker_data[f'sma_{fast_sma}'], label=f'{fast_sma}-Day SMA', color='blue', linestyle='--', alpha=0.7)
        ax.plot(ticker_data.index, ticker_data[f'sma_{slow_sma}'], label=f'{slow_sma}-Day SMA', color='orange', linestyle='--', alpha=0.7)

        # Plot buy signals
        buy_signals = buy_signals_df.get(f'{ticker}_signal')
        if buy_signals is not None:
            buy_dates = buy_signals[buy_signals == 1].index
            if not buy_dates.empty:
                buy_prices = ticker_data.loc[buy_dates]['close']
                ax.plot(buy_dates, buy_prices, '^', markersize=10, color='green', label='Buy Signal', linestyle='None')

        # Plot sell signals
        sell_signals = sell_signals_df.get(f'{ticker}_signal')
        if sell_signals is not None:
            sell_dates = sell_signals[sell_signals == 1].index
            if not sell_dates.empty:
                sell_prices = ticker_data.loc[sell_dates]['close']
                ax.plot(sell_dates, sell_prices, 'v', markersize=10, color='red', label='Sell Signal', linestyle='None')
            
        ax.set_title(f'{ticker} - Dual SMA Crossover Strategy')
        ax.set_ylabel('Price (USD)')
        ax.legend()
        ax.grid(True)

    for i in range(num_tickers, len(axes)):
        fig.delaxes(axes[i])

    fig.tight_layout()
    plt.savefig('momentum_strategy_plot.png')
    print("\nPlot saved as momentum_strategy_plot.png")

# --- New Momentum Strategy Logic ---

def generate_momentum_signals(data, fast_period=20, slow_period=50):
    """Generates buy signals based on a Dual Moving Average crossover."""
    
    # Calculate SMAs
    data[f'sma_{fast_period}'] = data['close'].rolling(window=fast_period).mean()
    data[f'sma_{slow_period}'] = data['close'].rolling(window=slow_period).mean()
    
    # Generate signal
    # A signal is 1 if the fast SMA crosses above the slow SMA
    data['signal'] = 0
    # The condition for a crossover
    condition = (data[f'sma_{fast_period}'] > data[f'sma_{slow_period}']) & \
                (data[f'sma_{fast_period}'].shift(1) <= data[f'sma_{slow_period}'].shift(1))
    
    data.loc[condition, 'signal'] = 1
    
    return data

from backtester import run_backtest

def generate_signals(data, fast_period=20, slow_period=50):
    """
    Generates buy and sell signals based on a simple Dual Moving Average crossover.
    """
    # --- Indicator Calculation ---
    data[f'sma_{fast_period}'] = data['close'].rolling(window=fast_period).mean()
    data[f'sma_{slow_period}'] = data['close'].rolling(window=slow_period).mean()
    
    # --- Buy Signal: Golden Cross ---
    buy_signals = (data[f'sma_{fast_period}'] > data[f'sma_{slow_period}']) & \
                  (data[f'sma_{fast_period}'].shift(1) <= data[f'sma_{slow_period}'].shift(1))
    
    # --- Sell Signal: Death Cross ---
    sell_signals = (data[f'sma_{fast_period}'] < data[f'sma_{slow_period}']) & \
                   (data[f'sma_{fast_period}'].shift(1) >= data[f'sma_{slow_period}'].shift(1))
                
    return pd.DataFrame({
        'buy_signal': buy_signals.astype(int),
        'sell_signal': sell_signals.astype(int)
    })

def main(tickers, fast_sma, slow_sma):
    """Main function to run the momentum trading strategy."""
    print("Starting the Momentum (Dual SMA Crossover) trading system...")

    # --- 1. Data Retrieval ---
    all_data = load_or_fetch_data(tickers, Client.KLINE_INTERVAL_1DAY, "5 years ago UTC")
    
    if not all_data:
        print("Could not fetch or load any data.")
        return

    print("\n--- Generating Momentum Buy & Sell Signals ---")
    
    all_buy_signals = {}
    all_sell_signals = {}
    total_buy_signals = 0
    total_sell_signals = 0

    for ticker in tickers:
        print(f"Processing {ticker}...")
        ticker_data = all_data[ticker]
        
        signals = generate_signals(ticker_data, fast_sma, slow_sma)
        
        # Add all generated data (SMAs, signals) to the main DataFrame
        all_data[ticker] = pd.concat([ticker_data, signals], axis=1)
        
        all_buy_signals[f'{ticker}_signal'] = signals['buy_signal']
        all_sell_signals[f'{ticker}_signal'] = signals['sell_signal']
        
        num_buy = signals['buy_signal'].sum()
        num_sell = signals['sell_signal'].sum()
        total_buy_signals += num_buy
        total_sell_signals += num_sell
        print(f"Generated {num_buy} buy and {num_sell} sell signals for {ticker}.")

    print(f"\nTotal buy signals: {total_buy_signals} | Total sell signals: {total_sell_signals}")

    # --- 2. Plotting ---
    buy_signals_df = pd.DataFrame(all_buy_signals)
    sell_signals_df = pd.DataFrame(all_sell_signals)
    plot_data_with_smas(all_data, buy_signals_df, sell_signals_df, tickers, fast_sma, slow_sma)

    # --- 3. Backtesting ---
    trade_log = run_backtest(all_data, buy_signals_df, sell_signals_df)

    print("--- Trade Log Sample ---")
    print(trade_log.head())
    print("------------------------")


if __name__ == "__main__":
    # Import the backtester
    from backtester import run_backtest
    
    # List of tickers to analyze
    tickers_to_process = ["BTCUSDT", "XRPUSDT", "ETHUSDT", "SOLUSDT", "DOGEUSDT", "ADAUSDT", 
                          "SHIBUSDT", "DOTUSDT", "BTTUSDT", "LINKUSDT", "ALGOUSDT", "AVAXUSDT"]
    # Define the periods for the moving averages
    fast_moving_avg = 20
    slow_moving_avg = 50
    
    main(tickers_to_process, fast_moving_avg, slow_moving_avg)
