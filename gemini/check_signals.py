import pandas as pd
from binance.client import Client
import os

import sys
sys.path.append("..")
sys.path.append(".")
# This script reuses functions from momentum_strategy.py
from gemini.momentum_strategy import load_or_fetch_data, generate_signals

def check_for_signals(tickers, fast_sma, slow_sma):
    """
    Checks for the latest buy and sell signals for a list of tickers.
    """
    print("Checking for buy and sell signals...")

    all_data = load_or_fetch_data(tickers, Client.KLINE_INTERVAL_1DAY, "100 days ago UTC")

    if not all_data:
        print("Could not fetch or load any data. Exiting.")
        return

    buy_signals_found = []
    sell_signals_found = []

    for ticker in tickers:
        if ticker not in all_data:
            print(f"No data for {ticker}, skipping.")
            continue

        ticker_data = all_data[ticker]
        
        # Generate buy and sell signals
        signals = generate_signals(ticker_data, fast_sma, slow_sma)
        
        # Get the last row for signal checking
        last_row = ticker_data.iloc[-1]
        last_date = last_row.name.date()
        last_close = last_row['close']

        # Check if the latest signal is a buy signal
        if signals['buy_signal'].iloc[-1] == 1:
            signal_info = {
                "ticker": ticker,
                "date": last_date,
                "close": last_close
            }
            print(f"BUY SIGNAL DETECTED FOR: {ticker} on {last_date} at close price {last_close}")
            buy_signals_found.append(signal_info)
        
        # Check if the latest signal is a sell signal
        if signals['sell_signal'].iloc[-1] == 1:
            signal_info = {
                "ticker": ticker,
                "date": last_date,
                "close": last_close
            }
            print(f"SELL SIGNAL DETECTED FOR: {ticker} on {last_date} at close price {last_close}")
            sell_signals_found.append(signal_info)

    if not buy_signals_found and not sell_signals_found:
        print("No new signals detected for any tickers.")
    else:
        if buy_signals_found:
            print("\n--- Summary of Tickers with Buy Signals ---")
            for signal in buy_signals_found:
                print(f"  - {signal['ticker']}: Close Price on {signal['date']} was ${signal['close']:.4f}")
            print("-------------------------------------------")
        
        if sell_signals_found:
            print("\n--- Summary of Tickers with Sell Signals ---")
            for signal in sell_signals_found:
                print(f"  - {signal['ticker']}: Close Price on {signal['date']} was ${signal['close']:.4f}")
            print("--------------------------------------------")

if __name__ == "__main__":
    # List of tickers to analyze
    tickers_to_process = ["BTCUSDT", "XRPUSDT", "ETHUSDT", "SOLUSDT", "DOGEUSDT", "ADAUSDT", 
                            "SHIBUSDT", "DOTUSDT", "BTTUSDT", "LINKUSDT", "ALGOUSDT", "AVAXUSDT",
                            "XLMUSDT", "NEARUSDT", "LTCUSDT", "CHZUSDT", "POLUSDT", "GRTUSDT",
                            "LRCUSDT", "ARBUSDT", "UNIUSDT", "GALAUSDT", "INJUSDT", "TRXUSDT", "CRVUSDT",
                            "ANKRUSDT", "NMRUSDT", "WOOUSDT", "MANAUSDT", "AAVEUSDT", "QNTUSDT", "BCHUSDT",
                            "SUSHIUSDT", "APEUSDT", "ZRXUSDT", "ETCUSDT", "KSMUSDT", "SANDUSDT", "IMXUSDT", 
                            "1INCHUSDT", "OPUSDT", "ATOMUSDT", "POWRUSDT", "AXSUSDT", "YFIUSDT", 
                            "SNXUSDT", "MKRUSDT", "STORJUSDT", "GNOUSDT", "BATUSDT", "REQUSDT", "COMPUSDT", 
                            "XTZUSDT", "BNTUSDT", "ENJUSDT", "EOSUSDT"]
    
    # Define the periods for the moving averages
    fast_moving_avg = 20
    slow_moving_avg = 50

    check_for_signals(tickers_to_process, fast_moving_avg, slow_moving_avg)

