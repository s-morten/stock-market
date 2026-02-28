"""
Unit tests for reit_dashboard.data.models (SQLAlchemy ORM).

Verifies table creation, basic constraints, and relationship behaviour
using an in-memory SQLite database (provided via conftest fixtures).
"""

from datetime import date
from decimal import Decimal

import pytest
from sqlalchemy import inspect

from reit_dashboard.data.models import Company, FinancialFact


class TestTableCreation:
    """Verify that the ORM models map to correctly named tables."""

    def test_companies_table_exists(self, db_engine):
        """The 'companies' table must be created by Base.metadata."""
        inspector = inspect(db_engine)
        assert "companies" in inspector.get_table_names()

    def test_financial_facts_table_exists(self, db_engine):
        """The 'financial_facts' table must be created by Base.metadata."""
        inspector = inspect(db_engine)
        assert "financial_facts" in inspector.get_table_names()


class TestCompanyModel:
    """Tests for the Company ORM model."""

    def test_insert_and_query(self, db_session):
        """A Company can be inserted and retrieved by PK."""
        company = Company(
            cik="0001045609",
            name="Prologis, Inc.",
            sic="6798",
            fiscal_year_end="12-31",
        )
        db_session.add(company)
        db_session.commit()

        fetched = db_session.get(Company, "0001045609")
        assert fetched is not None
        assert fetched.name == "Prologis, Inc."

    def test_optional_fields_accept_none(self, db_session):
        """sic and fiscal_year_end are nullable and should accept None."""
        company = Company(cik="0000000001", name="Test REIT")
        db_session.add(company)
        db_session.commit()

        fetched = db_session.get(Company, "0000000001")
        assert fetched.sic is None
        assert fetched.fiscal_year_end is None

    def test_repr(self):
        """__repr__ should include CIK and name."""
        company = Company(cik="0001045609", name="Prologis, Inc.")
        assert "0001045609" in repr(company)
        assert "Prologis" in repr(company)


class TestFinancialFactModel:
    """Tests for the FinancialFact ORM model."""

    def _make_company(self, session, cik: str = "0001045609") -> Company:
        company = Company(cik=cik, name="Test Corp")
        session.add(company)
        session.flush()
        return company

    def test_insert_and_query(self, db_session):
        """A FinancialFact can be inserted and retrieved."""
        self._make_company(db_session)
        fact = FinancialFact(
            cik="0001045609",
            concept="Revenues",
            period_end=date(2022, 12, 31),
            form="10-K",
            value=Decimal("5000000.00"),
            unit="USD",
        )
        db_session.add(fact)
        db_session.commit()

        fetched = db_session.get(FinancialFact, fact.id)
        assert fetched is not None
        assert fetched.concept == "Revenues"
        assert fetched.value == Decimal("5000000.00")

    def test_unique_constraint(self, db_session):
        """Inserting duplicate (cik, concept, period_end, form) raises."""
        from sqlalchemy.exc import IntegrityError

        self._make_company(db_session)
        kwargs = dict(
            cik="0001045609",
            concept="Revenues",
            period_end=date(2022, 12, 31),
            form="10-K",
            value=Decimal("5000000.00"),
            unit="USD",
        )
        db_session.add(FinancialFact(**kwargs))
        db_session.commit()

        db_session.add(FinancialFact(**kwargs))
        with pytest.raises(IntegrityError):
            db_session.commit()

    def test_relationship_to_company(self, db_session):
        """FinancialFact.company should navigate to the parent Company."""
        company = self._make_company(db_session)
        fact = FinancialFact(
            cik=company.cik,
            concept="Assets",
            period_end=date(2022, 12, 31),
            form="10-K",
            value=Decimal("100000000.00"),
            unit="USD",
        )
        db_session.add(fact)
        db_session.commit()
        db_session.refresh(fact)

        assert fact.company.name == "Test Corp"
