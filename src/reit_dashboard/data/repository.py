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

from reit_dashboard.data.models import Company, FinancialFact, MacroFact, PropertyFact, StockPrice


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

    def get_latest_period_end(self, cik: str) -> date | None:
        """
        Return the most recent ``period_end`` stored for a given CIK.

        Parameters:
            cik: Company CIK.

        Returns:
            date | None: Latest period end date, or ``None`` if no facts exist.
        """
        from sqlalchemy import func

        stmt = select(func.max(FinancialFact.period_end)).where(
            FinancialFact.cik == cik
        )
        return self._session.scalars(stmt).first()


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

    def get_latest_date(self, ticker: str) -> date | None:
        """
        Return the most recent date stored for a given ticker.

        Parameters:
            ticker: Exchange ticker symbol.

        Returns:
            date | None: Latest stored date, or ``None`` if no prices exist.
        """
        from sqlalchemy import func

        stmt = select(func.max(StockPrice.date)).where(
            StockPrice.ticker == ticker
        )
        return self._session.scalars(stmt).first()


class PropertyFactRepository:
    """
    Handles persistence for PropertyFact records.

    Parameters:
        session: An active SQLAlchemy Session.
    """

    def __init__(self, session: Session) -> None:
        self._session = session

    def upsert(self, row: dict) -> None:
        """
        Insert or update a single property fact row.

        Matches on the unique constraint (cik, accn, property_type,
        metric_name).  On conflict the ``value`` column is updated.

        Accepts ``period_end`` as either a :class:`datetime.date` or an
        ISO-format string; converts strings automatically.

        Parameters:
            row: Dict with keys matching PropertyFact columns.
        """
        # Normalise period_end to a date object (SQLite Date type requires it).
        row = dict(row)
        pe = row.get("period_end")
        if isinstance(pe, str) and pe:
            try:
                from datetime import date as _date
                row["period_end"] = _date.fromisoformat(pe)
            except ValueError:
                row["period_end"] = None
        elif pe == "":
            row["period_end"] = None

        stmt = (
            sqlite_insert(PropertyFact)
            .values(**row)
            .on_conflict_do_update(
                index_elements=["cik", "accn", "property_type", "metric_name"],
                set_={"value": row["value"]},
            )
        )
        self._session.execute(stmt)

    def upsert_many(self, rows: list[dict]) -> int:
        """
        Upsert a batch of property fact rows.

        Parameters:
            rows: List of dicts as returned by
                  :func:`~reit_dashboard.data.property_parser.property_result_to_rows`.

        Returns:
            int: Number of rows processed.
        """
        for row in rows:
            self.upsert(row)
        return len(rows)

    def get_facts(
        self,
        cik: str,
        form: str | None = None,
        metric_name: str | None = None,
    ) -> list[PropertyFact]:
        """
        Query property facts for a company with optional filters.

        Parameters:
            cik:         Company CIK.
            form:        Restrict to a specific form type (e.g. "10-K").
            metric_name: Restrict to a specific normalised metric key.

        Returns:
            list[PropertyFact]: Matching rows ordered by period_end.
        """
        stmt = (
            select(PropertyFact)
            .where(PropertyFact.cik == cik)
            .order_by(PropertyFact.period_end, PropertyFact.property_type)
        )
        if form is not None:
            stmt = stmt.where(PropertyFact.form == form)
        if metric_name is not None:
            stmt = stmt.where(PropertyFact.metric_name == metric_name)
        return list(self._session.scalars(stmt))

    def get_facts_for_ciks(
        self,
        ciks: list[str],
        metric_name: str | None = None,
        form: str | None = None,
    ) -> list[PropertyFact]:
        """
        Query property facts for multiple companies.

        Parameters:
            ciks:        List of CIKs.
            metric_name: Optional metric filter.
            form:        Optional form type filter.

        Returns:
            list[PropertyFact]: Matching rows.
        """
        stmt = (
            select(PropertyFact)
            .where(PropertyFact.cik.in_(ciks))
            .order_by(PropertyFact.cik, PropertyFact.period_end)
        )
        if metric_name is not None:
            stmt = stmt.where(PropertyFact.metric_name == metric_name)
        if form is not None:
            stmt = stmt.where(PropertyFact.form == form)
        return list(self._session.scalars(stmt))

    def get_latest_filing_accns(self, ciks: list[str]) -> dict[str, str]:
        """
        Return the most recent accession number per CIK that has data.

        Parameters:
            ciks: List of CIKs.

        Returns:
            dict: {cik: accn} for the most recent filed record per company.
        """
        result: dict[str, str] = {}
        for cik in ciks:
            stmt = (
                select(PropertyFact.accn)
                .where(PropertyFact.cik == cik)
                .order_by(PropertyFact.period_end.desc())
                .limit(1)
            )
            row = self._session.scalars(stmt).first()
            if row is not None:
                result[cik] = row
        return result

    def get_known_accns(self, cik: str) -> set[str]:
        """
        Return the set of all accession numbers already stored for a CIK.

        Used to skip filings that have already been fully processed so the
        ingestion pipeline is idempotent.

        Parameters:
            cik: Company CIK.

        Returns:
            set[str]: All distinct accession numbers in the database for
                this company.
        """
        stmt = (
            select(PropertyFact.accn)
            .where(PropertyFact.cik == cik)
            .distinct()
        )
        return set(self._session.scalars(stmt))


class MacroFactRepository:
    """
    Repository for MacroFact persistence and queries.

    Parameters:
        session: Active SQLAlchemy session.
    """

    def __init__(self, session: Session) -> None:
        self._session = session

    def upsert(self, fact: MacroFact) -> None:
        """
        Insert or update a single MacroFact.

        Uses SQLite's ``ON CONFLICT DO UPDATE`` to avoid duplicates on
        (series_id, date).

        Parameters:
            fact: MacroFact instance to persist.
        """
        stmt = (
            sqlite_insert(MacroFact)
            .values(
                series_id=fact.series_id,
                series_name=fact.series_name,
                date=fact.date,
                value=fact.value,
                unit=fact.unit,
                frequency=fact.frequency,
            )
            .on_conflict_do_update(
                index_elements=["series_id", "date"],
                set_={
                    "series_name": fact.series_name,
                    "value": fact.value,
                    "unit": fact.unit,
                    "frequency": fact.frequency,
                },
            )
        )
        self._session.execute(stmt)

    def get_series(
        self,
        series_id: str,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> list[MacroFact]:
        """
        Fetch all observations for one FRED series, optionally filtered by date.

        Parameters:
            series_id:  FRED series identifier.
            start_date: Earliest date (inclusive).
            end_date:   Latest date (inclusive).

        Returns:
            list[MacroFact]: Sorted by date ascending.
        """
        stmt = (
            select(MacroFact)
            .where(MacroFact.series_id == series_id)
            .order_by(MacroFact.date)
        )
        if start_date is not None:
            stmt = stmt.where(MacroFact.date >= start_date)
        if end_date is not None:
            stmt = stmt.where(MacroFact.date <= end_date)
        return list(self._session.scalars(stmt))

    def get_all_series(
        self,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> list[MacroFact]:
        """
        Fetch all stored macro observations, optionally filtered by date.

        Parameters:
            start_date: Earliest date (inclusive).
            end_date:   Latest date (inclusive).

        Returns:
            list[MacroFact]: Sorted by series_id and date.
        """
        stmt = (
            select(MacroFact)
            .order_by(MacroFact.series_id, MacroFact.date)
        )
        if start_date is not None:
            stmt = stmt.where(MacroFact.date >= start_date)
        if end_date is not None:
            stmt = stmt.where(MacroFact.date <= end_date)
        return list(self._session.scalars(stmt))

    def get_latest_date(self, series_id: str) -> date | None:
        """
        Return the most recent observation date stored for a FRED series.

        Parameters:
            series_id: FRED series identifier.

        Returns:
            date | None: Latest stored date, or ``None`` if no data exists.
        """
        from sqlalchemy import func

        stmt = select(func.max(MacroFact.date)).where(
            MacroFact.series_id == series_id
        )
        return self._session.scalars(stmt).first()
