import yfinance as yf
from database_io import DB_handler, Prices
from datetime import date

class PriceFetch:
    def __init__(self, db_handler: DB_handler):
        self.db_handler = db_handler

    def all_prices(self):
        # get all krz from db < 90 days old
        all_targets = self.db_handler.get_targets(days_cutoff=90)
        for target in all_targets:
            target_price = self._get_price(target.krz)
            target_krz = target.krz
            target_id = target.target_id
            today = date.today().strftime("%Y-%m-%d")
            self._price_to_db(target_price, target_krz, target_id, today)

    def _get_price(self, krz):
        try:
            ticker = yf.Ticker(f"{krz}")
            price_value = ticker.history(period="1d").Close.values[0]
            return price_value
        except:
            return None
    
        
    def _price_to_db(self, target_price, target_krz, target_id, today):
        price = Prices()
        price.price = target_price
        price.krz = target_krz
        price.target_id = target_id
        price.date = today
        self.db_handler.add_price(price)