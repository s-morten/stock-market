import pandas as pd
from binance.client import Client
import os

# This script reuses functions from momentum_strategy.py
from momentum_strategy import load_or_fetch_data, generate_signals

def check_for_buy_signals(tickers, fast_sma, slow_sma):
    """
    Checks for the latest buy signals for a list of tickers.
    """
    print("Checking for buy signals...")

    all_data = load_or_fetch_data(tickers, Client.KLINE_INTERVAL_1DAY, "100 days ago UTC")

    if not all_data:
        print("Could not fetch or load any data. Exiting.")
        return

    buy_signals_found = []

    for ticker in tickers:
        if ticker not in all_data:
            print(f"No data for {ticker}, skipping.")
            continue

        ticker_data = all_data[ticker]
        
        # Generate buy and sell signals
        signals = generate_signals(ticker_data, fast_sma, slow_sma)
        
        # Check if the latest signal is a buy signal
        if signals['buy_signal'].iloc[-1] == 1:
            print(f"BUY SIGNAL DETECTED FOR: {ticker}")
            buy_signals_found.append(ticker)

    if not buy_signals_found:
        print("No new buy signals detected for any tickers.")
    else:
        print("\n--- Summary of Tickers with Buy Signals ---")
        for ticker in buy_signals_found:
            print(ticker)
        print("-------------------------------------------")

if __name__ == "__main__":
    # List of tickers to analyze
    tickers_to_process = ["BTCUSDT", "XRPUSDT", "ETHUSDT", "SOLUSDT", "DOGEUSDT", "ADAUSDT", 
                          "SHIBUSDT", "DOTUSDT", "BTTUSDT", "LINKUSDT", "ALGOUSDT", "AVAXUSDT"]
    
    # Define the periods for the moving averages
    fast_moving_avg = 20
    slow_moving_avg = 50

    check_for_buy_signals(tickers_to_process, fast_moving_avg, slow_moving_avg)
