from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import time
import pandas as pd
import requests
from bs4 import BeautifulSoup
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import random
from database_io import DB_handler
from datetime import datetime, date, timedelta
from currency_converter import CurrencyConverter

c = CurrencyConverter(fallback_on_missing_rate=True)
dbh = DB_handler()


def currency_transform(val, currency, target_date):
    if currency == 'GBX':
        val = val / 100
        currency = 'GBP'
    return c.convert(val, currency, 'USD', date=target_date)

end_date = date.today()
start_date = date(2023, 4, 13)
day_count = (end_date - start_date).days

driver = webdriver.Chrome()  # Use appropriate driver (e.g., Firefox, Edge)
driver.get("https://www.marketbeat.com/ratings/")  # Replace with the actual URL

for single_date in (start_date + timedelta(n) for n in range(day_count)):
    scrape_date = datetime.strftime(single_date, "%m/%d/%Y")

    # Locate the input field (adjust the selector as needed)
    input_element = driver.find_element(By.ID, "cphPrimaryContent_txtStartDate")  # Or use NAME, XPATH, etc.

    driver.execute_script(f"""
        var input = arguments[0];
        input.value = '{scrape_date}';  // Change to the desired date
        input.dispatchEvent(new Event('change', {{ bubbles: true }}));  // Fire change event
    """, input_element)

    # Scrape the updated table
    soup = BeautifulSoup(driver.page_source, "html.parser")
    site_date = soup.find("input", {"id": "cphPrimaryContent_txtStartDate"}).get('value')
    while site_date != scrape_date:
        time.sleep(1)
        soup = BeautifulSoup(driver.page_source, "html.parser")
        site_date = soup.find("input", {"id": "cphPrimaryContent_txtStartDate"}).get('value')

    soup = BeautifulSoup(driver.page_source, "html.parser")
    table = soup.find("div", {"id": "cphPrimaryContent_pnlUpdate"})

    data = []
    if table:
        rows = table.find_all("tr")
        # Loop through rows and extract data
        for row in rows:
            # Extract table data from each row
            cells = row.find_all(["td"])
            if (not cells) or (len(cells) < 8):
                continue
            # Name
            divs = cells[0].find_all("div")
            krz = str(divs[-2].text)
            name = str(divs[-1].text)
            # current price
            
            broker = str(cells[2].get('data-sort-value')) if bool(cells[2].text) else None
            # target = float(cells[5].get('data-sort-value')) if bool(cells[5].text) else None
            # get target from text
            target_text = cells[5].getText()
            if target_text is None or target_text == '': 
                continue

            if '➝' in target_text:
                target_text = target_text.split('➝')[1]
                    
            other_currencys = {"C$": "CAD", 
                               "€": "EUR", 
                               "GBX": "GBX", 
                               "£": "GBP", 
                               "$": "USD",
                               "CHF": "CHF",}
        
            target = None
            for sub in other_currencys.keys():
                if sub in target_text:
                    target_text = target_text.replace(sub, '')
                    target_text = target_text.replace(',', '')
                    target_text = target_text.strip()
                    target_text = float(target_text)
                    target = currency_transform(target_text, other_currencys[sub], single_date)
                    break

            if None in [target, broker, scrape_date, krz, name, other_currencys[sub], target_text]:
                continue

            # Append extracted data to list
            data.append((single_date, krz, target, broker, name, other_currencys[sub], target_text))

    # Convert to DataFrame
    dbh.analyses.insert_target_bunch(data)
    time.sleep(random.randint(1, 6))

    # Close the driver
driver.quit()