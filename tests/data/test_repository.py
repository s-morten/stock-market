"""
Unit tests for reit_dashboard.data.repository.

Uses an in-memory SQLite database (provided via conftest fixtures).
"""

from datetime import date
from decimal import Decimal

import pytest

from reit_dashboard.data.models import Company, FinancialFact, StockPrice
from reit_dashboard.data.repository import (
    CompanyRepository,
    FinancialFactRepository,
    StockPriceRepository,
)


def _company(cik: str = "0001045609", name: str = "Test Corp") -> Company:
    return Company(cik=cik, name=name, sic="6798", fiscal_year_end="12-31")


def _fact(
    cik: str = "0001045609",
    concept: str = "Revenues",
    period_end: date = date(2022, 12, 31),
    form: str = "10-K",
    value: Decimal = Decimal("5000000.00"),
) -> FinancialFact:
    return FinancialFact(
        cik=cik, concept=concept, period_end=period_end, form=form,
        value=value, unit="USD",
    )


# ---------------------------------------------------------------------------
# CompanyRepository
# ---------------------------------------------------------------------------


class TestCompanyRepository:
    """Tests for CompanyRepository CRUD operations."""

    def test_upsert_insert(self, db_session):
        """upsert on a new CIK inserts the record."""
        repo = CompanyRepository(db_session)
        repo.upsert(_company())
        db_session.commit()

        assert repo.get_by_cik("0001045609") is not None

    def test_upsert_update(self, db_session):
        """upsert on an existing CIK updates the record."""
        repo = CompanyRepository(db_session)
        repo.upsert(_company(name="Old Name"))
        db_session.commit()

        repo.upsert(_company(name="New Name"))
        db_session.commit()

        fetched = repo.get_by_cik("0001045609")
        assert fetched.name == "New Name"

    def test_get_by_cik_not_found(self, db_session):
        """get_by_cik returns None when the CIK is not in the database."""
        repo = CompanyRepository(db_session)
        assert repo.get_by_cik("9999999999") is None

    def test_list_all_empty(self, db_session):
        """list_all returns an empty list when no companies are stored."""
        repo = CompanyRepository(db_session)
        assert repo.list_all() == []

    def test_list_all_ordered_by_name(self, db_session):
        """list_all returns companies sorted alphabetically by name."""
        repo = CompanyRepository(db_session)
        repo.upsert(_company("0000000002", "Zebra REIT"))
        repo.upsert(_company("0000000001", "Alpha REIT"))
        db_session.commit()

        names = [c.name for c in repo.list_all()]
        assert names == ["Alpha REIT", "Zebra REIT"]


# ---------------------------------------------------------------------------
# FinancialFactRepository
# ---------------------------------------------------------------------------


class TestFinancialFactRepository:
    """Tests for FinancialFactRepository CRUD operations."""

    def _seed_company(self, session, cik: str = "0001045609") -> None:
        session.add(_company(cik=cik))
        session.flush()

    def test_upsert_inserts_new_fact(self, db_session):
        """upsert on a new fact inserts a row."""
        self._seed_company(db_session)
        repo = FinancialFactRepository(db_session)
        repo.upsert(_fact())
        db_session.commit()

        facts = repo.get_facts("0001045609")
        assert len(facts) == 1

    def test_upsert_updates_existing_fact(self, db_session):
        """upsert with the same natural key updates the value."""
        self._seed_company(db_session)
        repo = FinancialFactRepository(db_session)
        repo.upsert(_fact(value=Decimal("1000.00")))
        db_session.commit()

        repo.upsert(_fact(value=Decimal("9999.00")))
        db_session.commit()

        facts = repo.get_facts("0001045609")
        assert len(facts) == 1
        assert facts[0].value == Decimal("9999.00")

    def test_get_facts_filter_by_concept(self, db_session):
        """get_facts with concept filter returns only matching rows."""
        self._seed_company(db_session)
        repo = FinancialFactRepository(db_session)
        repo.upsert(_fact(concept="Revenues"))
        repo.upsert(_fact(concept="Assets", period_end=date(2021, 12, 31)))
        db_session.commit()

        result = repo.get_facts("0001045609", concept="Revenues")
        assert len(result) == 1
        assert result[0].concept == "Revenues"

    def test_get_facts_filter_by_date_range(self, db_session):
        """get_facts respects start_date and end_date filters."""
        self._seed_company(db_session)
        repo = FinancialFactRepository(db_session)
        repo.upsert(_fact(period_end=date(2020, 12, 31)))
        repo.upsert(_fact(concept="Assets", period_end=date(2022, 12, 31)))
        db_session.commit()

        result = repo.get_facts(
            "0001045609",
            start_date=date(2021, 1, 1),
            end_date=date(2023, 1, 1),
        )
        assert len(result) == 1
        assert result[0].period_end == date(2022, 12, 31)

    def test_get_facts_empty_for_unknown_cik(self, db_session):
        """get_facts returns empty list for a CIK with no data."""
        repo = FinancialFactRepository(db_session)
        assert repo.get_facts("9999999999") == []

    def test_get_all_facts_no_filter(self, db_session):
        """get_all_facts returns facts across multiple companies."""
        self._seed_company(db_session, "0001045609")
        self._seed_company(db_session, "0000726854")
        repo = FinancialFactRepository(db_session)
        repo.upsert(_fact(cik="0001045609"))
        repo.upsert(_fact(cik="0000726854"))
        db_session.commit()

        result = repo.get_all_facts()
        assert len(result) == 2


def _stock_price(
    ticker: str = "PLD",
    dt: date = date(2024, 1, 1),
    close: Decimal = Decimal("100.00"),
) -> StockPrice:
    return StockPrice(
        ticker=ticker,
        date=dt,
        open=Decimal("98.00"),
        high=Decimal("105.00"),
        low=Decimal("97.00"),
        close=close,
        volume=1_000_000,
    )


class TestStockPriceRepository:
    """Tests for StockPriceRepository CRUD operations."""

    def test_upsert_inserts_new_price(self, db_session):
        """upsert on a new (ticker, date) inserts the row."""
        repo = StockPriceRepository(db_session)
        repo.upsert(_stock_price())
        db_session.commit()

        result = repo.get_prices("PLD")
        assert len(result) == 1

    def test_upsert_updates_existing_price(self, db_session):
        """upsert with the same (ticker, date) updates the close value."""
        repo = StockPriceRepository(db_session)
        repo.upsert(_stock_price(close=Decimal("100.00")))
        db_session.commit()

        repo.upsert(_stock_price(close=Decimal("115.00")))
        db_session.commit()

        result = repo.get_prices("PLD")
        assert len(result) == 1
        assert result[0].close == Decimal("115.00")

    def test_get_prices_empty_for_unknown_ticker(self, db_session):
        """get_prices returns [] when ticker has no data."""
        repo = StockPriceRepository(db_session)
        assert repo.get_prices("UNKNOWN") == []

    def test_get_prices_filter_by_date_range(self, db_session):
        """start_date and end_date filters are applied correctly."""
        repo = StockPriceRepository(db_session)
        repo.upsert(_stock_price(dt=date(2020, 1, 1)))
        repo.upsert(_stock_price(dt=date(2023, 6, 1)))
        repo.upsert(_stock_price(dt=date(2024, 1, 1)))
        db_session.commit()

        result = repo.get_prices(
            "PLD",
            start_date=date(2023, 1, 1),
            end_date=date(2023, 12, 31),
        )
        assert len(result) == 1
        assert result[0].date == date(2023, 6, 1)

    def test_get_prices_ordered_by_date(self, db_session):
        """Results are ordered by date ascending."""
        repo = StockPriceRepository(db_session)
        repo.upsert(_stock_price(dt=date(2024, 3, 1)))
        repo.upsert(_stock_price(dt=date(2024, 1, 1)))
        repo.upsert(_stock_price(dt=date(2024, 2, 1)))
        db_session.commit()

        result = repo.get_prices("PLD")
        dates = [r.date for r in result]
        assert dates == sorted(dates)

    def test_get_prices_for_tickers_multi_ticker(self, db_session):
        """get_prices_for_tickers returns data for all requested tickers."""
        repo = StockPriceRepository(db_session)
        repo.upsert(_stock_price(ticker="PLD"))
        repo.upsert(_stock_price(ticker="O"))
        repo.upsert(_stock_price(ticker="SPG"))
        db_session.commit()

        result = repo.get_prices_for_tickers(["PLD", "O"])
        tickers = {r.ticker for r in result}
        assert tickers == {"PLD", "O"}
        assert len(result) == 2
