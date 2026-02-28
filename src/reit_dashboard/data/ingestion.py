"""
Data ingestion service for REIT Dashboard.

Orchestrates the end-to-end pipeline:
  1. Fetch company metadata from EDGAR.
  2. Fetch XBRL quarterly (10-Q) financial facts for the last 5 years.
  3. Fetch weekly stock prices via Yahoo Finance for the last 5 years.
  4. Parse Item 2 property tables from 10-K/10-Q filing HTML.
  5. Use Gemini to extract total property counts from the tables.
  6. Validate and transform via Pydantic models.
  7. Persist to the database via repository layer.

This module owns the business logic; it depends on EdgarClient,
StockPriceClient, GeminiPropertyExtractor, and the repository classes
but does not know about HTTP or SQL details directly.
"""

import logging
from datetime import date
from decimal import Decimal
from typing import TYPE_CHECKING

from sqlalchemy.orm import Session

from reit_dashboard.data.edgar_client import EdgarClient
from reit_dashboard.data.models import Company, FinancialFact, StockPrice
from reit_dashboard.data.property_parser import (
    extract_item2_tables_text,
    extract_property_data,
    property_result_to_rows,
)
from reit_dashboard.data.repository import (
    CompanyRepository,
    FinancialFactRepository,
    PropertyFactRepository,
    StockPriceRepository,
)
from reit_dashboard.data.stock_price_client import StockPriceClient

if TYPE_CHECKING:
    from reit_dashboard.data.gemini_client import GeminiPropertyExtractor

logger = logging.getLogger(__name__)

# CIK → display-name mapping for the PoC REIT universe.
POC_REITS: dict[str, str] = {
    "0001045609": "Prologis",
    "0000726854": "Realty Income",
    "0001063761": "Simon Property Group",
    "0001393311": "Public Storage",
    "0000766704": "Welltower",
}

# CIK → exchange ticker symbol mapping.
POC_TICKERS: dict[str, str] = {
    "0001045609": "PLD",
    "0000726854": "O",
    "0001063761": "SPG",
    "0001393311": "PSA",
    "0000766704": "WELL",
}

# Only quarterly filings are ingested (the dashboard focuses on quarterly data).
_INGEST_FORMS: set[str] = {"10-Q"}


def _five_years_ago() -> date:
    """
    Return the date exactly 5 years before today.

    Returns:
        date: today's date with the year decremented by 5.
    """
    today = date.today()
    return today.replace(year=today.year - 5)


def ingest_company(
    cik: str,
    client: EdgarClient,
    session: Session,
    since_date: date | None = None,
) -> dict[str, int]:
    """
    Ingest one REIT company: metadata + quarterly financial facts.

    Parameters:
        cik:        Company CIK (raw or zero-padded).
        client:     Configured EdgarClient instance.
        session:    Active SQLAlchemy session (not yet committed).
        since_date: Earliest period end date to fetch.  Defaults to 5
                    years ago so ingestion is bounded to a rolling window.

    Returns:
        dict: Summary with keys "facts_upserted" and "concepts_fetched".
    """
    if since_date is None:
        since_date = _five_years_ago()

    company_repo = CompanyRepository(session)
    fact_repo = FinancialFactRepository(session)

    # --- 1. Company metadata (including ticker) ---
    info = client.fetch_company_info(cik)
    company = Company(
        cik=info.cik,
        name=info.name,
        sic=info.sic,
        fiscal_year_end=info.fiscal_year_end,
        ticker=POC_TICKERS.get(cik),
    )
    company_repo.upsert(company)

    # --- 2. Financial facts (10-Q, last 5 years) ---
    all_concept_facts = client.fetch_all_poc_facts(
        cik, since_date=since_date, forms=_INGEST_FORMS
    )
    facts_upserted = 0

    for concept_facts in all_concept_facts:
        for entry in concept_facts.entries:
            fact = FinancialFact(
                cik=info.cik,
                concept=concept_facts.concept,
                period_end=entry.end,
                form=entry.form,
                value=Decimal(str(entry.val)),
                unit=concept_facts.unit,
            )
            fact_repo.upsert(fact)
            facts_upserted += 1

    return {
        "facts_upserted": facts_upserted,
        "concepts_fetched": len(all_concept_facts),
    }


def ingest_all_poc_reits(
    client: EdgarClient,
    session: Session,
) -> list[dict[str, object]]:
    """
    Run ingestion for all companies in the PoC REIT universe.

    Commits after each company so a failure on one does not roll back
    data already persisted for previous companies.

    Parameters:
        client:  Configured EdgarClient instance.
        session: Active SQLAlchemy session.

    Returns:
        list[dict]: One summary dict per company with keys:
            "cik", "name", "facts_upserted", "concepts_fetched", "error".
    """
    results: list[dict[str, object]] = []

    for cik, name in POC_REITS.items():
        try:
            summary = ingest_company(cik, client, session)
            session.commit()
            results.append({"cik": cik, "name": name, "error": None, **summary})
        except Exception as exc:  # noqa: BLE001
            session.rollback()
            results.append(
                {
                    "cik": cik,
                    "name": name,
                    "facts_upserted": 0,
                    "concepts_fetched": 0,
                    "error": str(exc),
                }
            )

    return results


def ingest_stock_prices(
    ticker: str,
    stock_client: StockPriceClient,
    session: Session,
    since_date: date | None = None,
) -> dict[str, object]:
    """
    Ingest weekly stock prices for a single ticker.

    Parameters:
        ticker:       Exchange ticker symbol (e.g. "PLD").
        stock_client: Configured StockPriceClient instance.
        session:      Active SQLAlchemy session (not yet committed).
        since_date:   Earliest date to include. Defaults to 5 years ago.

    Returns:
        dict: Summary with keys "ticker" and "prices_upserted".
    """
    if since_date is None:
        since_date = _five_years_ago()

    price_repo = StockPriceRepository(session)
    entries = stock_client.fetch_weekly_prices(ticker, since_date=since_date)

    for entry in entries:
        price = StockPrice(
            ticker=ticker,
            date=entry.date,
            open=Decimal(str(entry.open)),
            high=Decimal(str(entry.high)),
            low=Decimal(str(entry.low)),
            close=Decimal(str(entry.close)),
            volume=entry.volume,
        )
        price_repo.upsert(price)

    return {"ticker": ticker, "prices_upserted": len(entries)}


def ingest_all_stock_prices(
    stock_client: StockPriceClient,
    session: Session,
) -> list[dict[str, object]]:
    """
    Run weekly stock price ingestion for all PoC REIT tickers.

    Commits after each ticker so a failure on one does not roll back
    data already persisted for previous tickers.

    Parameters:
        stock_client: Configured StockPriceClient instance.
        session:      Active SQLAlchemy session.

    Returns:
        list[dict]: One summary dict per ticker with keys:
            "cik", "name", "ticker", "prices_upserted", "error".
    """
    results: list[dict[str, object]] = []

    for cik, name in POC_REITS.items():
        ticker = POC_TICKERS.get(cik)
        if ticker is None:
            results.append(
                {
                    "cik": cik,
                    "name": name,
                    "ticker": None,
                    "prices_upserted": 0,
                    "error": "No ticker mapping defined",
                }
            )
            continue

        try:
            summary = ingest_stock_prices(ticker, stock_client, session)
            session.commit()
            results.append(
                {"cik": cik, "name": name, "error": None, **summary}
            )
        except Exception as exc:  # noqa: BLE001
            session.rollback()
            results.append(
                {
                    "cik": cik,
                    "name": name,
                    "ticker": ticker,
                    "prices_upserted": 0,
                    "error": str(exc),
                }
            )

    return results


def ingest_property_data(
    cik: str,
    ticker: str | None,
    client: EdgarClient,
    session: Session,
    since_date: date | None = None,
    forms: set[str] | None = None,
    max_filings: int = 10,
    gemini_extractor: "GeminiPropertyExtractor | None" = None,
) -> dict[str, object]:
    """
    Ingest Item 2 property table data for one company.

    For each filing in the requested window the method:
    1. Lists filings from the EDGAR submissions endpoint.
    2. Fetches the primary HTML document for the filing.
    3. Extracts all tables from the Item 2 "Properties" section.
    4. (Optional) Sends the table text to Gemini to extract the total
       property count and stores it as a ``PropertyFact`` row with
       ``property_type="total"`` and ``metric_name="total_properties_gemini"``.
    5. Persists the extracted rows via :class:`PropertyFactRepository`.

    Parameters:
        cik:               Company CIK (raw or zero-padded).
        ticker:            Exchange ticker symbol (for denormalisation).
        client:            Configured :class:`EdgarClient` instance.
        session:           Active SQLAlchemy session (not yet committed).
        since_date:        Earliest filing date to include.
                           Defaults to 5 years ago.
        forms:             Form types to fetch.  Defaults to ``{"10-K"}``.
        max_filings:       Maximum number of filings to process.
        gemini_extractor:  Optional :class:`GeminiPropertyExtractor`.
                           When ``None`` the Gemini step is skipped.

    Returns:
        dict: Summary with keys ``filings_processed``, ``rows_upserted``,
        ``filings_parsed``, ``gemini_counts`` (number of Gemini extractions).
    """
    if since_date is None:
        since_date = _five_years_ago()
    if forms is None:
        # Only 10-K filings for property data – they contain the most
        # complete Item 2 description.
        forms = {"10-K"}

    prop_repo = PropertyFactRepository(session)

    filings = client.list_filings(
        cik, forms=forms, since_date=since_date, max_filings=max_filings
    )

    filings_processed = 0
    filings_parsed = 0
    rows_upserted = 0
    gemini_counts = 0

    for filing in filings:
        accn = filing["accn"]
        form = filing["form"]
        period_end = filing.get("reportDate") or filing.get("filingDate", "")

        try:
            html = client.fetch_filing_html(
                cik, accn, primary_doc=filing.get("primaryDocument") or None
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not fetch HTML for %s / %s: %s", cik, accn, exc)
            filings_processed += 1
            continue

        filings_processed += 1

        # --- Gemini property count extraction ---
        if gemini_extractor is not None:
            tables_text = extract_item2_tables_text(html)
            if tables_text:
                try:
                    gemini_result = gemini_extractor.extract_property_count(tables_text)
                    count = gemini_result.get("total_properties")
                    notes = gemini_result.get("notes", "")
                    logger.info(
                        "Gemini [%s/%s]: total_properties=%s – %s",
                        cik, accn, count, notes,
                    )
                    if count is not None:
                        rows_upserted += prop_repo.upsert_many([{
                            "cik": cik,
                            "ticker": ticker,
                            "accn": accn,
                            "period_end": period_end,
                            "form": form,
                            "property_type": "total",
                            "metric_name": "total_properties_gemini",
                            "value": str(count),
                        }])
                        gemini_counts += 1
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "Gemini extraction failed for %s / %s: %s", cik, accn, exc
                    )
            else:
                logger.debug("No Item 2 tables found for Gemini in %s / %s", cik, accn)

        # --- HTML table row extraction (property-type breakdown) ---
        result = extract_property_data(html, accn, form)
        if result.records:
            db_rows = property_result_to_rows(result, cik, ticker, period_end)
            rows_upserted += prop_repo.upsert_many(db_rows)
            filings_parsed += 1
        else:
            logger.debug("No property table found in %s / %s", cik, accn)

    return {
        "filings_processed": filings_processed,
        "filings_parsed": filings_parsed,
        "rows_upserted": rows_upserted,
        "gemini_counts": gemini_counts,
    }


def ingest_all_property_data(
    client: EdgarClient,
    session: Session,
    since_date: date | None = None,
    max_filings_per_company: int = 10,
    gemini_extractor: "GeminiPropertyExtractor | None" = None,
) -> list[dict[str, object]]:
    """
    Run property data ingestion for all companies in the PoC universe.

    Commits after each company so a failure on one does not roll back
    data already persisted for previous companies.

    Parameters:
        client:                  Configured :class:`EdgarClient` instance.
        session:                 Active SQLAlchemy session.
        since_date:              Earliest filing date to include.
        max_filings_per_company: Cap per company to limit latency.
        gemini_extractor:        Optional :class:`GeminiPropertyExtractor`.
                                 When provided, Gemini is used to extract
                                 total property counts from Item 2 tables.

    Returns:
        list[dict]: One summary dict per company.
    """
    results: list[dict[str, object]] = []

    for cik, name in POC_REITS.items():
        ticker = POC_TICKERS.get(cik)
        try:
            summary = ingest_property_data(
                cik,
                ticker,
                client,
                session,
                since_date=since_date,
                max_filings=max_filings_per_company,
                gemini_extractor=gemini_extractor,
            )
            session.commit()
            results.append({"cik": cik, "name": name, "ticker": ticker, "error": None, **summary})
        except Exception as exc:  # noqa: BLE001
            session.rollback()
            results.append(
                {
                    "cik": cik,
                    "name": name,
                    "ticker": ticker,
                    "filings_processed": 0,
                    "filings_parsed": 0,
                    "rows_upserted": 0,
                    "gemini_counts": 0,
                    "error": str(exc),
                }
            )

    return results
