from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy import Column, Integer, String, Float
from sqlalchemy.ext.declarative import declarative_base
from datetime import date, timedelta


class Targets(declarative_base()):
    __tablename__ = "targets"

    name = Column(String)
    krz = Column(String)
    start = Column(Float)
    date = Column(String)
    rating = Column(Integer)
    target = Column(Float)
    target_id = Column(Integer, primary_key=True)
    recommender = Column(String)

class Prices(declarative_base()):
    __tablename__ = "prices"

    price_id = Column(Integer, primary_key=True)
    target_id = Column(Integer)
    krz = Column(String)
    price = Column(Float)
    date = Column(String)

class DB_handler:
    def __init__(self, db_path):
        self.engine = create_engine(f'sqlite:///{db_path}', echo=False)
        self.session = Session(self.engine)

    def add_target(self, target: Targets):
        self.session.add(target)
        self.session.commit()

    def get_targets(self, days_cutoff=90):
        cuttoff_date = (date.today() - timedelta(days=days_cutoff)).strftime("%Y-%m-%d")
        return self.session.query(Targets).filter(Targets.date >= cuttoff_date).all()
    
    def add_price(self, price: Prices):
        self.session.add(price)
        self.session.commit()

