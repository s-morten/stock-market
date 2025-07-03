import pandas as pd
from binance.client import Client
import os
import numpy as np
import matplotlib.pyplot as plt
import math

# --- Configuration ---
# It's recommended to use environment variables for API keys
api_key = os.environ.get('BINANCE_API_KEY')
api_secret = os.environ.get('BINANCE_API_SECRET')

# Or, uncomment and hardcode them (not recommended for production)
# api_key = "YOUR_API_KEY"
# api_secret = "YOUR_API_SECRET"

client = Client(api_key, api_secret)

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
        
        # Convert necessary columns to float
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

def calculate_atr(df, period=14):
    """Calculates the Average True Range (ATR)."""
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr

def plot_data(data, combined_signals_df, tickers):
    """Plots the price data and buy signals for multiple tickers."""
    num_tickers = len(tickers)
    num_cols = 2
    num_rows = math.ceil(num_tickers / num_cols)
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(14, 5 * num_rows), sharex=True)
    axes = axes.flatten()

    for i, ticker in enumerate(tickers):
        ax = axes[i]
        ticker_data = data[ticker]
        ax.plot(ticker_data.index, ticker_data['close'], label=f'{ticker} Price')
        
        ticker_signal_col = f'{ticker}_signal'
        if ticker_signal_col in combined_signals_df.columns:
            signals = combined_signals_df[ticker_signal_col]
            buy_signals_dates = signals[signals == 1].index
            
            if not buy_signals_dates.empty:
                buy_prices = ticker_data.loc[buy_signals_dates]['close']
                ax.plot(buy_signals_dates, buy_prices, '^', markersize=8, color='g', label='Buy Signal', linestyle='None')
            
        ax.set_title(f'{ticker} Price and Buy Signals')
        ax.set_ylabel('Price (USD)')
        ax.legend()
        ax.grid(True)

    for i in range(num_tickers, len(axes)):
        fig.delaxes(axes[i])

    fig.tight_layout()
    plt.savefig('crypto_trading_plot.png')
    print("\nPlot saved as crypto_trading_plot.png")

def calculate_volume_weighted_average(returns, volumes):
    """Calculates the volume-weighted average return, falling back to a simple average if total volume is zero."""
    valid_returns = returns.dropna()
    if valid_returns.empty:
        return np.nan
    valid_volumes = volumes.loc[valid_returns.index]
    
    if valid_volumes.sum() == 0:
        return valid_returns.mean()
    else:
        return np.average(valid_returns, weights=valid_volumes)

def get_dynamic_gain_period(market_volatility, low_thresh, high_thresh):
    """Determines the gain period based on market volatility."""
    if market_volatility > high_thresh:
        return 3 # High volatility -> short period
    elif market_volatility < low_thresh:
        return 7 # Low volatility -> long period
    else:
        return 5 # Normal volatility

def main(tickers):
    """Main function to run the trading logic."""
    print("Starting the crypto trading system...")

    # --- 1. Data Retrieval ---
    all_data = load_or_fetch_data(tickers, Client.KLINE_INTERVAL_1DAY, "1 year ago UTC")
    
    if not all_data:
        print("Could not fetch or load any data.")
        return

    # --- 2. Feature Engineering ---
    print("\n--- Engineering Features (SMA, ATR, etc.) ---")
    combined_data = pd.concat(all_data, axis=1)
    
    for ticker in tickers:
        combined_data[(ticker, 'returns')] = combined_data[(ticker, 'close')].pct_change()
        combined_data[(ticker, 'atr14')] = calculate_atr(combined_data[ticker], period=14)
        combined_data[(ticker, 'sma50')] = combined_data[(ticker, 'close')].rolling(window=50).mean()

    # --- Create Volume-Weighted Market Benchmark & Volatility ---
    returns_df = combined_data.xs('returns', level=1, axis=1)
    volume_df = combined_data.xs('volume', level=1, axis=1)
    atr_df = combined_data.xs('atr14', level=1, axis=1)
    
    combined_data[('market', 'combined_returns')] = returns_df.apply(
        lambda x: calculate_volume_weighted_average(x, volume_df.loc[x.name]), axis=1)
    combined_data[('market', 'volatility')] = atr_df.mean(axis=1) # Average ATR as market vol
    
    combined_data.dropna(inplace=True)

    # --- 3. Signal Generation ---
    print("\n--- Generating Signals with Adaptive & Correlated Criteria ---")
    
    rolling_window = 50 # Increased to account for SMA
    
    # Define volatility thresholds for dynamic gain period
    low_vol_thresh = combined_data[('market', 'volatility')].quantile(0.25)
    high_vol_thresh = combined_data[('market', 'volatility')].quantile(0.75)

    potential_signals = {} # Store potential signals for each day: {date: [tickers]}

    for i in range(rolling_window, len(combined_data)):
        current_date = combined_data.index[i]
        gain_period = get_dynamic_gain_period(combined_data[('market', 'volatility')].iloc[i], low_vol_thresh, high_vol_thresh)
        
        if i < rolling_window + gain_period:
            continue

        # --- Market Expectation (Monte Carlo) ---
        historical_market_returns = combined_data[('market', 'combined_returns')].iloc[i-rolling_window:i]
        mean_return = historical_market_returns.mean()
        std_dev = historical_market_returns.std()

        if std_dev == 0 or np.isnan(std_dev): continue
        
        num_simulations = 1000
        simulated_period_returns = np.sum(
            np.random.normal(mean_return, std_dev, (num_simulations, gain_period)), axis=1)

        for ticker in tickers:
            # --- Rule 1: Long-Term Trend Filter ---
            if combined_data[(ticker, 'close')].iloc[i] < combined_data[(ticker, 'sma50')].iloc[i]:
                continue

            # --- Rule 2: Volume Confirmation ---
            signal_period_volume = combined_data[(ticker, 'volume')].iloc[i-gain_period:i].mean()
            lookback_period_volume = combined_data[(ticker, 'volume')].iloc[i-rolling_window:i].mean()
            if lookback_period_volume == 0 or signal_period_volume < lookback_period_volume * 1.2:
                continue

            # --- Rule 3: Volatility Filter ---
            current_atr = combined_data[(ticker, 'atr14')].iloc[i]
            avg_atr = combined_data[(ticker, 'atr14')].iloc[i-rolling_window:i].mean()
            if avg_atr == 0 or current_atr > avg_atr * 2.0:
                continue

            # --- Rule 4: Pullback Entry Proxy ---
            high = combined_data[(ticker, 'high')].iloc[i]
            low = combined_data[(ticker, 'low')].iloc[i]
            close = combined_data[(ticker, 'close')].iloc[i]
            if high == low or (high - close) / (high - low) < 0.1:
                continue

            # --- Rule 5: Sustained Gain Analysis ---
            actual_period_returns = combined_data[(ticker, 'returns')].iloc[i-gain_period:i]
            total_gain = actual_period_returns.sum()
            if total_gain <= 0 or actual_period_returns.max() / total_gain > 0.60:
                continue

            # --- Rule 6: Statistical Significance ---
            worse_simulations = np.sum(simulated_period_returns < total_gain)
            p_value = worse_simulations / len(simulated_period_returns)

            if p_value > 0.95:
                if current_date not in potential_signals:
                    potential_signals[current_date] = []
                potential_signals[current_date].append(ticker)

    # --- 4. Correlation Filtering and Final Signal Confirmation ---
    for ticker in tickers:
        combined_data[(ticker, 'signal')] = 0

    correlation_window = 30
    returns_df = combined_data.xs('returns', level=1, axis=1)
    
    for date, candidates in potential_signals.items():
        if len(candidates) == 1:
            # If only one signal, it's automatically confirmed
            loc_index = combined_data.index.get_loc(date)
            combined_data.iloc[loc_index, combined_data.columns.get_loc((candidates[0], 'signal'))] = 1
        else:
            # If multiple signals, find the least correlated one
            start_loc = combined_data.index.get_loc(date) - correlation_window
            end_loc = combined_data.index.get_loc(date)
            
            corr_matrix = returns_df.iloc[start_loc:end_loc][candidates].corr()
            avg_correlations = corr_matrix.mean(axis=1)
            least_correlated_ticker = avg_correlations.idxmin()
            
            loc_index = combined_data.index.get_loc(date)
            combined_data.iloc[loc_index, combined_data.columns.get_loc((least_correlated_ticker, 'signal'))] = 1

    total_signals = int(combined_data.xs('signal', level=1, axis=1).sum().sum())
    print(f"\nTotal confirmed signals generated after all filters: {total_signals}")

    # --- 5. Plotting ---
    plotting_data_reformatted = {ticker: combined_data[ticker] for ticker in tickers}
    signals_df = combined_data.xs('signal', level=1, axis=1)
    signals_df.columns = [f"{c}_signal" for c in signals_df.columns]

    plot_data(plotting_data_reformatted, signals_df, tickers)


if __name__ == "__main__":
    tickers_to_process = ["BTCUSDT", "XRPUSDT", "ETHUSDT", "SOLUSDT", "DOGEUSDT", "ADAUSDT"]
    main(tickers_to_process)
