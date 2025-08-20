import yfinance as yf
import pandas as pd
from database_io import DB_handler
from datetime import timedelta
import time

def get_friday_closes(ticker: str, start: str, end: str):
    """
    Fetches the closing prices of a given stock on Fridays within a specified date range.
    
    :param ticker: The stock ticker symbol (e.g., 'AAPL').
    :param start: The start date in 'YYYY-MM-DD' format.
    :param end: The end date in 'YYYY-MM-DD' format.
    :return: A Pandas Series with dates as index and closing prices as values.
    """
    # Download historical data with weekly interval (Friday closes)
    data = yf.download(ticker, start=start, end=end, interval="1wk")
    
    # Ensure the index is a datetime type
    data.index = pd.to_datetime(data.index)
    
    # Return only the closing prices
    return data['Close']

dbh = DB_handler()

targets_tuples = dbh.analyses.grouped_targets()
from tqdm import tqdm
for ticker, date_max, date_min in tqdm(targets_tuples, desc="Updating valuations"):
    date_max = date_max + timedelta(weeks=52)
    friday_closes = get_friday_closes(ticker, 
                                      date_min.strftime("%Y-%m-%d"), 
                                      date_max.strftime("%Y-%m-%d"))
    batch = [(ticker, date, close) for date, close in zip(friday_closes.index, friday_closes.values)]
    # print(batch)
    dbh.valuations.insert_valuation_batch(batch)
    time.sleep(1)
    # add to db
