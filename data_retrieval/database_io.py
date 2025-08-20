from sqlalchemy import create_engine
from sqlalchemy.orm import Session
import os
import oracledb
from sqlalchemy import Column, Integer, String, Date, Float, func
from sqlalchemy.ext.declarative import declarative_base
import datetime

class DB_handler_connection:
    def __init__(self):
        username = "ADMIN"
        password = "wg2fCHg9G5nRxPT"#os.environ.get("ORACLE_DB_PWD")
        dsn = "most"
        self.connection = oracledb.connect(user=username, password=password,
                            dsn=dsn, config_dir="/etc/")

        self.engine = create_engine('oracle+oracledb://', creator=lambda: self.connection)
        
        self.session = Session(self.engine)

class DB_handler():
    def __init__(self):
        connection = DB_handler_connection()
        self.analyses = Analyses_func(connection)
        self.valuations = Valuations_func(connection)

class DB_function():
    def __init__(self, connection_item):
        self.connection = connection_item.connection
        self.session = connection_item.session
        self.engine = connection_item.engine

class Analyses(declarative_base()):
    __tablename__ = "ANALYSES"
    __table_args__ = {'schema': 'STOCKS'}

    id = Column(Integer, primary_key=True)
    target_date = Column(Date)
    ticker = Column(String)
    target_usd = Column(Float)
    broker = Column(String)
    name = Column(String)
    currency = Column(String)
    target_orig = Column(Float)


class Analyses_func(DB_function):
    def insert_target_bunch(self, batch):
        targets = [Analyses(target_date=date, ticker=ticker, target_usd=target_usd, broker=broker, name=name, currency=currency, target_orig=target_orig) 
                        for date, ticker, target_usd, broker, name, currency, target_orig in batch]
        self.session.bulk_save_objects(targets)
        self.session.commit()

    def grouped_targets(self):
        return self.session.query(Analyses.ticker, 
                                  func.max(Analyses.target_date).label('max_date'),
                                  func.min(Analyses.target_date).label('min_date')
                                  ).group_by(Analyses.ticker).all()

class Companies(declarative_base()):
    __tablename__ = "COMPANIES"
    __table_args__ = {'schema': 'STOCKS'}

    ticker = Column(String, primary_key=True)
    name = Column(String)
    country = Column(String)
    sector = Column(String)

class Companies_func(DB_function):
    def insert_company(self, batch):
        pass

class Valuations(declarative_base()):
    __tablename__ = "VALUATIONS"
    __table_args__ = {'schema': 'STOCKS'}

    ticker = Column(String, primary_key=True)
    val_date = Column(Date, primary_key=True)
    valuation = Column(Float)

class Valuations_func(DB_function):
    def insert_valuation_batch(self, batch):
        valuations = [Valuations(ticker=ticker, val_date=date, valuation=value) for ticker, date, value in batch]
        self.session.bulk_save_objects(valuations)
        self.session.commit()

    def analysis_valuations(self, offset):
        return (self.session.query(Analyses.id, Analyses.target_date, Analyses.ticker, Analyses.broker, Analyses.target_usd, 
                                   Valuations.valuation)
                .filter(Valuations.ticker == Analyses.ticker)
                .filter(Valuations.val_date >  Analyses.target_date + datetime.timedelta(days=offset*7))
                .filter(Valuations.val_date <= Analyses.target_date + datetime.timedelta(days=(offset+1)*7))
                .all()
        )

class Comp(declarative_base()):
    __tablename__ = "COMPS"
    __table_args__ = {'schema': 'STOCKS'}

    ticker = Column(String, primary_key=True)
    sector_country = Column(String)
    date = Column(Date)
    value = Column(Float)
    pe  = Column(Float)

class Comp_func(DB_function):
        
    def insert_comp_batch(self, batch):
        pass

