import pandas as pd
from binance.client import Client
import os
import numpy as np
import matplotlib.pyplot as plt
import math
from backtester import run_backtest

# --- Configuration ---
client = Client()

# --- Data Fetching and Caching ---
def get_historical_data(symbol, interval, lookback):
    """Fetches historical kline data from Binance."""
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
    """Loads data from a CSV cache if valid, otherwise fetches from Binance."""
    data_cols = ['open', 'high', 'low', 'close', 'volume']
    required_cache_cols = [f"{t}_{c}" for t in tickers for c in data_cols]

    try:
        if os.path.exists(cache_filename):
            cached_df = pd.read_csv(cache_filename, index_col='timestamp', parse_dates=True)
            missing_cols = [c for c in required_cache_cols if c not in cached_df.columns]
            is_stale = (cached_df.index.max().date() < (pd.Timestamp.utcnow() - pd.Timedelta(days=1)).date()) if not cached_df.empty else False

            if not missing_cols and not is_stale:
                print("Cache is valid. Loading data from cache.")
                all_data = {t: cached_df[[f"{t}_{c}" for c in data_cols]].rename(columns=lambda c: c.replace(f"{t}_", "")) for t in tickers}
                return all_data
            else:
                print("Cache is invalid or stale. Fetching fresh data.")
    except Exception as e:
        print(f"Error with cache file: {e}. Fetching fresh data.")

    all_data_fetch = {t: get_historical_data(t, interval, lookback) for t in tickers}
    all_data_fetch = {t: d for t, d in all_data_fetch.items() if not d.empty}

    if not all_data_fetch:
        return {}

    combined_for_cache = pd.concat([df.rename(columns={c: f"{t}_{c}" for c in df.columns}) for t, df in all_data_fetch.items()], axis=1)
    combined_for_cache.to_csv(cache_filename)
    print(f"Data fetched and saved to {cache_filename}")
    
    return all_data_fetch

# --- Plotting ---
def plot_rebound_strategy(data, buy_signals_df, sell_signals_df, tickers):
    """Plots the price, volume, and buy/sell signals for the rebound strategy."""
    num_tickers = len(tickers)
    num_cols = 1
    num_rows = num_tickers * 2  # Price and Volume plots for each ticker
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(16, 6 * num_tickers), sharex=True)
    
    for i, ticker in enumerate(tickers):
        price_ax = axes[i*2]
        volume_ax = axes[i*2 + 1]
        ticker_data = data[ticker]
        
        # Plot price
        price_ax.plot(ticker_data.index, ticker_data['close'], label=f'{ticker} Price', color='black')
        price_ax.set_title(f'{ticker} - Rebound Strategy Analysis')
        price_ax.set_ylabel('Price (USD)')
        price_ax.grid(True)

        # Plot buy signals
        buy_signals = buy_signals_df.get(f'{ticker}_signal')
        if buy_signals is not None:
            buy_dates = buy_signals[buy_signals == 1].index
            if not buy_dates.empty:
                buy_prices = ticker_data.loc[buy_dates]['close']
                price_ax.plot(buy_dates, buy_prices, '^', markersize=12, color='green', label='Buy Signal')

        # Plot sell signals
        sell_signals = sell_signals_df.get(f'{ticker}_signal')
        if sell_signals is not None:
            sell_dates = sell_signals[sell_signals == 1].index
            if not sell_dates.empty:
                sell_prices = ticker_data.loc[sell_dates]['close']
                price_ax.plot(sell_dates, sell_prices, 'v', markersize=12, color='red', label='Sell Signal')
        
        price_ax.legend()

        # Plot volume
        volume_ax.bar(ticker_data.index, ticker_data['volume'], label=f'{ticker} Volume', color='grey', alpha=0.6)
        volume_ax.plot(ticker_data.index, ticker_data['volume_long_avg'], color='orange', linestyle='--', label='Long-term Avg Volume')
        volume_ax.set_ylabel('Volume')
        volume_ax.set_xlabel('Date')
        volume_ax.grid(True)
        volume_ax.legend()

    fig.tight_layout()
    plt.savefig('rebound_strategy_plot.png')
    print("\nPlot saved as rebound_strategy_plot.png")

# --- Rebound Strategy Logic ---
def generate_rebound_signals(data, loss_threshold=-0.10, lookback_period=7, short_vol_window=5, long_vol_window=50, volume_spike_factor=3, plateau_window=10, plateau_std_factor=0.5):
    """
    Generates buy signals based on a price drop, volume spike, and subsequent volume plateau.
    """
    # Calculate price change over the lookback period
    data['price_change'] = data['close'].pct_change(periods=lookback_period)
    
    # Calculate short and long-term moving averages for volume
    data['volume_short_avg'] = data['volume'].rolling(window=short_vol_window).mean()
    data['volume_long_avg'] = data['volume'].rolling(window=long_vol_window).mean()
    
    # --- Signal Generation ---
    data['signal'] = 0
    
    # Condition 1: Price drops significantly & Volume spikes
    condition_price_drop = data['price_change'] < loss_threshold
    condition_volume_spike = data['volume_short_avg'] > data['volume_long_avg'] * volume_spike_factor
    
    potential_rebound_dates = data[condition_price_drop & condition_volume_spike].index
    
    if potential_rebound_dates.empty:
        return data[['signal']] # Return empty signals if no potential rebounds found

    # Find where the volume subsides and plateaus after a spike
    for date in potential_rebound_dates:
        # Look in the future for the volume to calm down
        search_period = data.loc[date + pd.Timedelta(days=1) : date + pd.Timedelta(days=30)]
        
        if search_period.empty:
            continue

        # Condition 2: Volume returns to normal levels
        volume_calmed_down = search_period[search_period['volume_short_avg'] < search_period['volume_long_avg'] * 1.5]
        
        if not volume_calmed_down.empty:
            # Condition 3: Volume plateaus (low standard deviation)
            first_calm_date = volume_calmed_down.index[0]
            plateau_search_start = first_calm_date
            plateau_period = data.loc[plateau_search_start : plateau_search_start + pd.Timedelta(days=plateau_window)]
            
            if not plateau_period.empty:
                # Check if volume standard deviation is low compared to the long-term average
                plateau_volume_std = plateau_period['volume'].std()
                long_term_avg_vol_at_plateau = plateau_period['volume_long_avg'].iloc[0]
                
                if plateau_volume_std < (long_term_avg_vol_at_plateau * plateau_std_factor):
                    # Check to ensure we don't place a signal too far in the future from the initial drop
                    if (first_calm_date - date).days < 20:
                         # Place buy signal on the first day the volume has calmed down
                        data.loc[first_calm_date, 'signal'] = 1

    # Ensure we only have one signal per "event" by removing signals too close to each other
    signal_dates = data[data['signal'] == 1].index
    if len(signal_dates) > 1:
        to_remove = []
        last_signal_date = signal_dates[0]
        for current_signal_date in signal_dates[1:]:
            if (current_signal_date - last_signal_date).days < 30: # Cooldown period
                to_remove.append(current_signal_date)
            else:
                last_signal_date = current_signal_date
        data.loc[to_remove, 'signal'] = 0

    return data[['signal']]

def generate_trailing_stop_signals(data, buy_signals, trailing_percentage=0.05, min_hold_days=7):
    """
    Generates sell signals based on a trailing stop loss with a minimum holding period.
    """
    signals = pd.DataFrame(index=data.index)
    signals['sell_signal'] = 0
    
    in_trade = False
    peak_price_since_entry = 0
    entry_date = None
    
    for i in range(len(data)):
        # Check for a buy signal to enter a trade
        if buy_signals.iloc[i]['signal'] == 1 and not in_trade:
            in_trade = True
            entry_date = data.index[i]
            # Assume entry on the next day's open
            if i + 1 < len(data):
                entry_price = data.iloc[i+1]['open']
                peak_price_since_entry = entry_price
            else:
                # Cannot enter trade if it's the last day
                in_trade = False
                entry_date = None
        
        # If in a trade, check for stop loss condition
        if in_trade:
            current_high = data.iloc[i]['high']
            current_low = data.iloc[i]['low']
            
            # Update the peak price since entry
            peak_price_since_entry = max(peak_price_since_entry, current_high)
            
            # Only check for sell signal if min holding period has passed
            if (data.index[i] - entry_date).days >= min_hold_days:
                # Calculate the stop loss price
                stop_loss_price = peak_price_since_entry * (1 - trailing_percentage)
                
                # Check if the current low has breached the stop loss price
                if current_low < stop_loss_price:
                    signals.iloc[i]['sell_signal'] = 1
                    in_trade = False
                    peak_price_since_entry = 0 # Reset for next trade
                    entry_date = None
                
    return signals

def main(tickers, loss_threshold, lookback_period, short_vol, long_vol, vol_factor, plat_window, plat_factor, trailing_stop_pct, min_hold_days):
    """Main function to run the rebound trading strategy."""
    print("Starting the Rebound trading system...")

    # --- 1. Data Retrieval ---
    all_data = load_or_fetch_data(tickers, Client.KLINE_INTERVAL_1DAY, "5 years ago UTC")
    
    if not all_data:
        print("Could not fetch or load any data.")
        return

    print("\n--- Generating Rebound Buy & Sell Signals ---")
    
    all_buy_signals = {}
    all_sell_signals = {}
    total_buy_signals = 0
    total_sell_signals = 0

    for ticker in tickers:
        print(f"Processing {ticker}...")
        ticker_data = all_data[ticker]
        
        # Generate Buy Signals
        buy_signals = generate_rebound_signals(ticker_data, loss_threshold, lookback_period, short_vol, long_vol, vol_factor, plat_window, plat_factor)
        all_data[ticker] = pd.concat([ticker_data, buy_signals.rename(columns={'signal': 'buy_signal'})], axis=1)
        all_buy_signals[f'{ticker}_signal'] = buy_signals['signal']
        
        # Generate Sell Signals
        sell_signals = generate_trailing_stop_signals(ticker_data, buy_signals, trailing_stop_pct, min_hold_days)
        all_data[ticker] = pd.concat([all_data[ticker], sell_signals], axis=1)
        all_sell_signals[f'{ticker}_signal'] = sell_signals['sell_signal']

        num_buy = buy_signals['signal'].sum()
        num_sell = sell_signals['sell_signal'].sum()
        total_buy_signals += num_buy
        total_sell_signals += num_sell
        print(f"Generated {num_buy} buy and {num_sell} sell signals for {ticker}.")

    print(f"\nTotal buy signals: {total_buy_signals} | Total sell signals: {total_sell_signals}")

    # --- 2. Plotting ---
    buy_signals_df = pd.DataFrame(all_buy_signals)
    sell_signals_df = pd.DataFrame(all_sell_signals)
    plot_rebound_strategy(all_data, buy_signals_df, sell_signals_df, tickers)

    # --- 3. Backtesting ---
    trade_log = run_backtest(all_data, buy_signals_df, sell_signals_df)

    if not trade_log.empty:
        print("\n--- Trade Log Sample ---")
        print(trade_log.head())
        print("------------------------")

if __name__ == "__main__":
    tickers_to_process = ["BTCUSDT", "XRPUSDT", "ETHUSDT", "SOLUSDT", "DOGEUSDT", "ADAUSDT"]
    
    # --- Strategy Parameters ---
    loss_threshold = -0.10       # 20% price drop
    lookback_days = 7            # over 7 days
    short_vol_window = 7         # 5-day avg for volume spike detection
    long_vol_window = 35         # 60-day avg for baseline volume
    volume_spike_factor = 1.5      # Volume must be 3x the long-term average
    plateau_window = 3          # 10-day window to check for a plateau
    plateau_std_factor = 0.7     # Std dev of volume must be less than 0.7x the long-term avg
    trailing_stop_pct = 0.08     # 8% trailing stop loss
    min_hold_days = 7            # Minimum number of days to hold before stop loss is active
    
    main(
        tickers=tickers_to_process,
        loss_threshold=loss_threshold,
        lookback_period=lookback_days,
        short_vol=short_vol_window,
        long_vol=long_vol_window,
        vol_factor=volume_spike_factor,
        plat_window=plateau_window,
        plat_factor=plateau_std_factor,
        trailing_stop_pct=trailing_stop_pct,
        min_hold_days=min_hold_days
    )
