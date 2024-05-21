import requests
from bs4 import BeautifulSoup

from database_io import DB_handler, Targets
from datetime import datetime, date




class Scraper:
    def __init__(self, db_handler: DB_handler):
        self.recommodations_list = ["barclays-stock-recommendations", "berenberg-bank-stock-recommendations",
                                    "goldman-sachs-group-stock-recommendations"]
        self.db_handler = db_handler

    def scrape_table(self):
        targets = []
        # scrape page
        for website in self.recommodations_list:
            response = requests.get(f"https://www.marketbeat.com/ratings/by-issuer/{website}/")
            if response.status_code == 200:
                content = response.content
            else:
                print("Failed to fetch the page. Status code:", response.status_code)
                raise ValueError(f"Failed scraping {url}")
            
            soup = BeautifulSoup(content, "html.parser")
            table = soup.find("div", {"id": "cphPrimaryContent_pnlFilterTable"})
            if table:
                rows = table.find_all("tr")
                # Loop through rows and extract data
                for row in rows:
                    price_target = Targets()
                    # Extract table data from each row
                    cells = row.find_all(["td"])
                    if len(cells) < 10:
                        continue
                    # Name
                    divs = cells[0].find_all("div")
                    price_target.krz = str(divs[-2].text)
                    price_target.name = str(divs[-1].text)
                    # current price
                    price_target.start = float(cells[1].get('data-sort-value'))
                    price_target.date = str(date_compare:=datetime.strptime(cells[2].text, "%m/%d/%Y").date())
                    
                    if date_compare < date.today():
                        continue
                    price_target.rating = int(cells[6].get('data-sort-value'))
                    price_target.target = float(cells[7].get('data-sort-value')) if bool(cells[7].text) else None
                    price_target.recommender = str(website.rstrip("-stock-recommendations"))
                    targets.append(price_target)
            else:
                raise ValueError("Table not found on the page.")
        return targets

    def validate_table():
        pass

    def table_to_db(self, targets):
        for target in targets:
            self.db_handler.add_target(target)