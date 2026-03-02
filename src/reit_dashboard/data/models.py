"""
SQLAlchemy ORM models for the REIT Dashboard.

Three models are defined:
- Company:       static metadata for a REIT (CIK, name, SIC code, etc.)
- FinancialFact: a single XBRL financial data point for a company/period.
- StockPrice:    weekly OHLCV price bar for a publicly traded REIT.
- PropertyFact:  a single property metric parsed from a filing's Item 2.
- MacroFact:     a single macroeconomic data point from the FRED API.

No Oracle-specific dialect features are used; the same models work with
SQLite (development) and Oracle (production) by changing DATABASE_URL.
"""

from datetime import date
from decimal import Decimal

from sqlalchemy import (
    BigInteger,
    Date,
    ForeignKey,
    Integer,
    Numeric,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Shared declarative base for all ORM models."""

    pass


class Company(Base):
    """
    Represents a REIT company as registered with the SEC.

    Attributes:
        cik:             Central Index Key – unique SEC identifier (PK).
        name:            Legal company name.
        sic:             Standard Industrial Classification code.
        fiscal_year_end: Month and day the fiscal year ends (MM-DD).
        facts:           Relationship to associated FinancialFact rows.
    """

    __tablename__ = "companies"

    cik: Mapped[str] = mapped_column(String(20), primary_key=True)
    name: Mapped[str] = mapped_column(String(256), nullable=False)
    sic: Mapped[str | None] = mapped_column(String(10), nullable=True)
    fiscal_year_end: Mapped[str | None] = mapped_column(
        String(5), nullable=True
    )
    # Stock ticker symbol (e.g. "PLD").  Nullable because not all companies
    # tracked via EDGAR will have a matching publicly traded ticker.
    ticker: Mapped[str | None] = mapped_column(String(20), nullable=True)

    facts: Mapped[list["FinancialFact"]] = relationship(
        back_populates="company", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Company cik={self.cik!r} name={self.name!r}>"


class FinancialFact(Base):
    """
    A single XBRL financial fact for a company and reporting period.

    Each row represents one value (e.g. Revenues = 5 000 000 USD) for a
    specific company, concept, period, and filing form.  The unique
    constraint prevents duplicate ingestion of the same data point.

    Attributes:
        id:         Surrogate primary key.
        cik:        FK to Company.
        concept:    GAAP XBRL tag, e.g. "Revenues".
        period_end: End date of the reporting period.
        form:       SEC form type, e.g. "10-K" or "10-Q".
        value:      Reported numeric value.
        unit:       Unit of measure, typically "USD".
    """

    __tablename__ = "financial_facts"
    __table_args__ = (
        UniqueConstraint(
            "cik", "concept", "period_end", "form",
            name="uq_fact_cik_concept_period_form",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    cik: Mapped[str] = mapped_column(
        String(20), ForeignKey("companies.cik"), nullable=False
    )
    concept: Mapped[str] = mapped_column(String(128), nullable=False)
    period_end: Mapped[date] = mapped_column(Date, nullable=False)
    form: Mapped[str] = mapped_column(String(20), nullable=False)
    value: Mapped[Decimal] = mapped_column(Numeric(precision=20, scale=2), nullable=False)
    unit: Mapped[str] = mapped_column(String(20), nullable=False, default="USD")

    company: Mapped["Company"] = relationship(back_populates="facts")

    def __repr__(self) -> str:
        return (
            f"<FinancialFact cik={self.cik!r} concept={self.concept!r} "
            f"period_end={self.period_end!r} value={self.value!r}>"
        )


class StockPrice(Base):
    """
    A weekly OHLCV price bar for a publicly traded REIT.

    Prices are stored by ticker symbol so the table is independent of the
    EDGAR CIK scheme and can be extended to any equity without a filing.
    The unique constraint on (ticker, date) prevents duplicate ingestion.

    Attributes:
        id:     Surrogate primary key.
        ticker: Exchange ticker symbol (e.g. "PLD").
        date:   Week start date of the price bar.
        open:   Opening price (USD).
        high:   Intra-week high price (USD).
        low:    Intra-week low price (USD).
        close:  Adjusted closing price (USD, split- and dividend-adjusted).
        volume: Total shares traded during the week.
    """

    __tablename__ = "stock_prices"
    __table_args__ = (
        UniqueConstraint(
            "ticker", "date",
            name="uq_stock_price_ticker_date",
        ),
    )

    id: Mapped[int] = mapped_column(
        Integer, primary_key=True, autoincrement=True
    )
    ticker: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    date: Mapped[date] = mapped_column(Date, nullable=False)
    open: Mapped[Decimal] = mapped_column(
        Numeric(precision=12, scale=4), nullable=False
    )
    high: Mapped[Decimal] = mapped_column(
        Numeric(precision=12, scale=4), nullable=False
    )
    low: Mapped[Decimal] = mapped_column(
        Numeric(precision=12, scale=4), nullable=False
    )
    close: Mapped[Decimal] = mapped_column(
        Numeric(precision=12, scale=4), nullable=False
    )
    volume: Mapped[int] = mapped_column(BigInteger, nullable=False)

    def __repr__(self) -> str:
        return (
            f"<StockPrice ticker={self.ticker!r} date={self.date!r}"
            f" close={self.close!r}>"
        )


class PropertyFact(Base):
    """
    A single property metric extracted from a REIT filing's Item 2.

    Each row represents one (property_type, metric_name, value) triple
    from the property table found in a 10-K or 10-Q filing.

    Example row:
        cik="0001045609", property_type="Industrial", metric_name="num_properties",
        value="1200", form="10-K", period_end=date(2023, 12, 31)

    Attributes:
        id:            Surrogate primary key.
        cik:           FK to Company.
        ticker:        Ticker symbol (denormalised for convenience).
        accn:          SEC accession number of the source filing.
        period_end:    Reporting period end date (from filing reportDate).
        form:          Filing form type (e.g. "10-K").
        property_type: Row label from the table (e.g. "Office").
        metric_name:   Normalised metric key (e.g. "num_properties").
        value:         Raw string value as parsed (may contain "%" etc.).
    """

    __tablename__ = "property_facts"
    __table_args__ = (
        UniqueConstraint(
            "cik", "accn", "property_type", "metric_name",
            name="uq_property_fact",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    cik: Mapped[str] = mapped_column(
        String(20), ForeignKey("companies.cik"), nullable=False, index=True
    )
    ticker: Mapped[str | None] = mapped_column(String(20), nullable=True)
    accn: Mapped[str] = mapped_column(String(40), nullable=False)
    period_end: Mapped[date | None] = mapped_column(Date, nullable=True)
    form: Mapped[str] = mapped_column(String(20), nullable=False)
    property_type: Mapped[str] = mapped_column(String(256), nullable=False)
    metric_name: Mapped[str] = mapped_column(String(64), nullable=False)
    value: Mapped[str] = mapped_column(String(64), nullable=False)

    def __repr__(self) -> str:
        return (
            f"<PropertyFact cik={self.cik!r} period_end={self.period_end!r}"
            f" property_type={self.property_type!r} {self.metric_name}={self.value!r}>"
        )


class MacroFact(Base):
    """
    A single macroeconomic data point fetched from the FRED API.

    Each row represents one value for a specific FRED series on a specific
    date (e.g. FEDFUNDS = 5.33 on 2024-01-01).

    Attributes:
        id:          Surrogate primary key.
        series_id:   FRED series identifier (e.g. "FEDFUNDS").
        series_name: Human-readable name (e.g. "Federal Funds Rate").
        date:        Observation date.
        value:       Observed numeric value.
        unit:        Unit of measure (e.g. "Percent", "Thousands of Persons").
        frequency:   Reporting frequency (e.g. "Monthly", "Weekly").
    """

    __tablename__ = "macro_facts"
    __table_args__ = (
        UniqueConstraint(
            "series_id", "date",
            name="uq_macro_fact_series_date",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    series_id: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    series_name: Mapped[str] = mapped_column(String(256), nullable=False)
    date: Mapped[date] = mapped_column(Date, nullable=False)
    value: Mapped[Decimal] = mapped_column(Numeric(precision=20, scale=6), nullable=False)
    unit: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    frequency: Mapped[str] = mapped_column(String(32), nullable=False, default="")

    def __repr__(self) -> str:
        return (
            f"<MacroFact series={self.series_id!r} date={self.date!r}"
            f" value={self.value!r}>"
        )
