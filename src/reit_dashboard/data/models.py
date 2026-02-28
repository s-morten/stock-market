"""
SQLAlchemy ORM models for the REIT Dashboard.

Two models are defined:
- Company:       static metadata for a REIT (CIK, name, SIC code, etc.)
- FinancialFact: a single XBRL financial data point for a company/period.

No Oracle-specific dialect features are used; the same models work with
SQLite (development) and Oracle (production) by changing DATABASE_URL.
"""

from datetime import date
from decimal import Decimal

from sqlalchemy import (
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
