import database_io as db
import target_scraper as ts
import finance_api as fa


if __name__ == "__main__":
    dbh = db.DB_handler("/sm")
    ts = ts.Scraper(dbh)
    fa = fa.PriceFetch(dbh)
    targets = ts.scrape_table()
    ts.table_to_db(targets)
    fa.all_prices()