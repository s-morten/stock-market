"""
Shared pytest fixtures for the REIT Dashboard test suite.
"""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from reit_dashboard.data.models import Base


@pytest.fixture(scope="function")
def db_engine():
    """
    Provide a fresh in-memory SQLite engine for each test.

    Using an in-memory database ensures tests are isolated and fast.
    """
    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(engine)
    yield engine
    Base.metadata.drop_all(engine)
    engine.dispose()


@pytest.fixture(scope="function")
def db_session(db_engine) -> Session:
    """
    Provide a SQLAlchemy Session bound to the in-memory engine.

    The session is rolled back after every test to prevent state leakage.
    """
    factory = sessionmaker(bind=db_engine)
    session = factory()
    yield session
    session.rollback()
    session.close()
