
"""
Backtester Module

This module provides functionality to backtest trading strategies based on buy and sell signals.
It can be imported into other scripts to evaluate strategy performance.
"""

import pandas as pd
import numpy as np

def run_backtest(all_data, buy_signals, sell_signals=None, initial_capital=10000):
    """
    Runs a backtest on a given set of buy and optional sell signals.

    Args:
        all_data (dict): A dictionary where keys are ticker symbols and values are
                         pandas DataFrames containing OHLCV data.
        buy_signals (pd.DataFrame): A DataFrame with a datetime index and columns
                                    named '{ticker}_signal' containing 1 for a buy
                                    signal and 0 otherwise.
        sell_signals (pd.DataFrame, optional): A DataFrame with the same format as
                                               buy_signals for sell signals. If None,
                                               random exits between 10 and 50 days
                                               will be simulated. Defaults to None.
        initial_capital (float): The starting capital for the backtest simulation.

    Returns:
        pd.DataFrame: A DataFrame containing a detailed log of all simulated trades.
                      Returns an empty DataFrame if no trades are made.
    """
    print("\n--- Starting Backtest ---")
    trades = []
    
    tickers = [col.replace('_signal', '') for col in buy_signals.columns]

    for ticker in tickers:
        print(f"Backtesting {ticker}...")
        
        if ticker not in all_data or f'{ticker}_signal' not in buy_signals.columns:
            print(f"Warning: Skipping {ticker}, data or signal column not found.")
            continue
            
        price_data = all_data[ticker]
        buy_signal_dates = buy_signals[buy_signals[f'{ticker}_signal'] == 1].index

        if sell_signals is not None and f'{ticker}_signal' in sell_signals.columns:
            sell_signal_dates = sell_signals[sell_signals[f'{ticker}_signal'] == 1].index
            active_buy_date = None
            for date in price_data.index:
                if date in buy_signal_dates and active_buy_date is None:
                    active_buy_date = date
                
                if date in sell_signal_dates and active_buy_date is not None:
                    entry_date = active_buy_date + pd.Timedelta(days=1)
                    exit_date = date + pd.Timedelta(days=1)

                    if entry_date in price_data.index and exit_date in price_data.index:
                        entry_price = price_data.loc[entry_date]['high']
                        exit_price = price_data.loc[exit_date]['low']
                        
                        if entry_price > 0:
                            trade_return = (exit_price - entry_price) / entry_price
                            trades.append({
                                'ticker': ticker, 'entry_signal_date': active_buy_date,
                                'exit_signal_date': date, 'entry_price': entry_price,
                                'exit_price': exit_price, 'return': trade_return
                            })
                    active_buy_date = None
        else:
            for buy_date in buy_signal_dates:
                holding_period = np.random.randint(10, 51)
                entry_date = buy_date + pd.Timedelta(days=1)
                exit_date = buy_date + pd.Timedelta(days=holding_period)

                if entry_date in price_data.index and exit_date in price_data.index:
                    entry_price = price_data.loc[entry_date]['high']
                    exit_price = price_data.loc[exit_date]['low']
                    
                    if entry_price > 0:
                        trade_return = (exit_price - entry_price) / entry_price
                        trades.append({
                            'ticker': ticker, 'entry_signal_date': buy_date,
                            'exit_signal_date': exit_date, 'entry_price': entry_price,
                            'exit_price': exit_price, 'return': trade_return
                        })

    if not trades:
        print("Backtest complete. No trades were executed.")
        return pd.DataFrame()

    # --- Performance Analysis ---
    trades_df = pd.DataFrame(trades)
    trades_df = trades_df.sort_values(by='entry_signal_date').reset_index(drop=True)
    
    # --- Core Metrics ---
    total_trades = len(trades_df)
    winning_trades = (trades_df['return'] > 0).sum()
    win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
    average_return = trades_df['return'].mean()
    
    # --- Profitability Metrics ---
    gross_profit = trades_df[trades_df['return'] > 0]['return'].sum()
    gross_loss = trades_df[trades_df['return'] < 0]['return'].sum()
    profit_factor = abs(gross_profit / gross_loss) if gross_loss != 0 else np.inf

    # --- Equity Curve and Drawdown ---
    # Assume equal investment per trade for simplicity
    trades_df['equity_curve'] = initial_capital * (1 + trades_df['return']).cumprod()
    
    peak = trades_df['equity_curve'].cummax()
    drawdown = (trades_df['equity_curve'] - peak) / peak
    max_drawdown = drawdown.min()
    
    # --- Sharpe Ratio (annualized) ---
    # Assuming daily returns and 252 trading days in a year
    daily_returns = trades_df.set_index('exit_signal_date')['return']
    sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252) if daily_returns.std() != 0 else 0

    print("\n--- Backtest Performance Metrics ---")
    print(f"Total Trades Executed:      {total_trades}")
    print(f"Win Rate:                   {win_rate:.2f}%")
    print(f"Profit Factor:              {profit_factor:.2f}")
    print("---")
    print(f"Average Return per Trade:   {average_return * 100:.2f}%")
    print(f"Total Return on Capital:    {(trades_df['equity_curve'].iloc[-1] / initial_capital - 1) * 100:.2f}%")
    print("---")
    print(f"Max Drawdown:               {max_drawdown * 100:.2f}%")
    print(f"Sharpe Ratio (annualized):  {sharpe_ratio:.2f}")
    print("-------------------------------------\n")
    
    return trades_df

if __name__ == '__main__':
    print("This script is intended to be imported as a module.")
    print("It contains the `run_backtest` function for evaluating trading strategies.")

