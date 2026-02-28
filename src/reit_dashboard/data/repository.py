"""
Database repository layer for REIT Dashboard.

Provides session-scoped CRUD operations via SQLAlchemy.  Business logic
(validation, orchestration) lives in the ingestion module; repositories
only talk to the database.

Using the repository pattern isolates the rest of the code from the ORM
and makes swapping the underlying database (SQLite → Oracle) trivial.
"""

from datetime import date
from decimal import Decimal

from sqlalchemy import select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Session

from reit_dashboard.data.models import Company, FinancialFact, StockPrice


class CompanyRepository:
    """
    Handles persistence for Company records.

    Parameters:
        session: An active SQLAlchemy Session.
    """

    def __init__(self, session: Session) -> None:
        self._session = session

    def upsert(self, company: Company) -> None:
        """
        Insert or update a Company record.

        Uses a merge strategy: if a company with the same CIK already
        exists its fields are updated; otherwise a new row is inserted.

        Parameters:
            company: Company ORM instance to persist.
        """
        # Session.merge handles the "upsert by PK" pattern portably.
        self._session.merge(company)

    def get_by_cik(self, cik: str) -> Company | None:
        """
        Retrieve a Company by its CIK.

        Parameters:
            cik: SEC Central Index Key.

        Returns:
            Company | None: The matching company, or None if not found.
        """
        return self._session.get(Company, cik)

    def list_all(self) -> list[Company]:
        """
        Return all stored companies ordered by name.

        Returns:
            list[Company]: All Company rows.
        """
        stmt = select(Company).order_by(Company.name)
        return list(self._session.scalars(stmt))


class FinancialFactRepository:
    """
    Handles persistence for FinancialFact records.

    Parameters:
        session: An active SQLAlchemy Session.
    """

    def __init__(self, session: Session) -> None:
        self._session = session

    def upsert(self, fact: FinancialFact) -> None:
        """
        Insert or update a FinancialFact record.

        The unique constraint (cik, concept, period_end, form) determines
        whether this is an insert or an update.  Uses session.merge for
        dialect-portable behaviour.

        Parameters:
            fact: FinancialFact ORM instance to persist.
        """
        stmt = (
            sqlite_insert(FinancialFact)
            .values(
                cik=fact.cik,
                concept=fact.concept,
                period_end=fact.period_end,
                form=fact.form,
                value=fact.value,
                unit=fact.unit,
            )
            .on_conflict_do_update(
                index_elements=["cik", "concept", "period_end", "form"],
                set_={"value": fact.value, "unit": fact.unit},
            )
        )
        self._session.execute(stmt)

    def get_facts(
        self,
        cik: str,
        concept: str | None = None,
        form: str | None = None,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> list[FinancialFact]:
        """
        Query financial facts with optional filters.

        Parameters:
            cik:        Company CIK (required).
            concept:    Filter by XBRL concept name.
            form:       Filter by filing form type (e.g. "10-K").
            start_date: Include only facts with period_end >= start_date.
            end_date:   Include only facts with period_end <= end_date.

        Returns:
            list[FinancialFact]: Matching rows ordered by period_end.
        """
        stmt = select(FinancialFact).where(FinancialFact.cik == cik)

        if concept is not None:
            stmt = stmt.where(FinancialFact.concept == concept)
        if form is not None:
            stmt = stmt.where(FinancialFact.form == form)
        if start_date is not None:
            stmt = stmt.where(FinancialFact.period_end >= start_date)
        if end_date is not None:
            stmt = stmt.where(FinancialFact.period_end <= end_date)

        stmt = stmt.order_by(FinancialFact.period_end)
        return list(self._session.scalars(stmt))

    def get_all_facts(
        self,
        concept: str | None = None,
        form: str | None = None,
    ) -> list[FinancialFact]:
        """
        Return facts across all companies with optional filters.

        Parameters:
            concept: Filter by XBRL concept name.
            form:    Filter by filing form type.

        Returns:
            list[FinancialFact]: Matching rows ordered by cik, period_end.
        """
        stmt = select(FinancialFact)
        if concept is not None:
            stmt = stmt.where(FinancialFact.concept == concept)
        if form is not None:
            stmt = stmt.where(FinancialFact.form == form)
        stmt = stmt.order_by(FinancialFact.cik, FinancialFact.period_end)
        return list(self._session.scalars(stmt))


class StockPriceRepository:
    """
    Handles persistence for StockPrice records.

    Parameters:
        session: An active SQLAlchemy Session.
    """

    def __init__(self, session: Session) -> None:
        self._session = session

    def upsert(self, price: StockPrice) -> None:
        """
        Insert or update a weekly stock price bar.

        The unique constraint (ticker, date) determines whether this is
        an insert or an update of the close/ohlcv values.

        Parameters:
            price: StockPrice ORM instance to persist.
        """
        stmt = (
            sqlite_insert(StockPrice)
            .values(
                ticker=price.ticker,
                date=price.date,
                open=price.open,
                high=price.high,
                low=price.low,
                close=price.close,
                volume=price.volume,
            )
            .on_conflict_do_update(
                index_elements=["ticker", "date"],
                set_={
                    "open": price.open,
                    "high": price.high,
                    "low": price.low,
                    "close": price.close,
                    "volume": price.volume,
                },
            )
        )
        self._session.execute(stmt)

    def get_prices(
        self,
        ticker: str,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> list[StockPrice]:
        """
        Query weekly price bars for a single ticker.

        Parameters:
            ticker:     Exchange ticker symbol.
            start_date: Include only bars with date >= start_date.
            end_date:   Include only bars with date <= end_date.

        Returns:
            list[StockPrice]: Matching rows ordered by date ascending.
        """
        stmt = (
            select(StockPrice)
            .where(StockPrice.ticker == ticker)
            .order_by(StockPrice.date)
        )
        if start_date is not None:
            stmt = stmt.where(StockPrice.date >= start_date)
        if end_date is not None:
            stmt = stmt.where(StockPrice.date <= end_date)
        return list(self._session.scalars(stmt))

    def get_prices_for_tickers(
        self,
        tickers: list[str],
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> list[StockPrice]:
        """
        Query weekly price bars for multiple tickers at once.

        Parameters:
            tickers:    List of exchange ticker symbols.
            start_date: Include only bars with date >= start_date.
            end_date:   Include only bars with date <= end_date.

        Returns:
            list[StockPrice]: Matching rows ordered by ticker, date.
        """
        stmt = (
            select(StockPrice)
            .where(StockPrice.ticker.in_(tickers))
            .order_by(StockPrice.ticker, StockPrice.date)
        )
        if start_date is not None:
            stmt = stmt.where(StockPrice.date >= start_date)
        if end_date is not None:
            stmt = stmt.where(StockPrice.date <= end_date)
        return list(self._session.scalars(stmt))
