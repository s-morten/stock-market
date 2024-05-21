FROM python:3.10.4

RUN pip install pandas yfinance datetime sqlalchemy requests beautifulsoup4

ADD data_retrieval/ .

CMD ["python", "./main.py"]