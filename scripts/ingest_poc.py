"""
PoC ingestion script – seeds the database with data for 5 US REITs.

Fetches:
  - Company metadata + quarterly financial facts from SEC EDGAR
  - Weekly stock prices from Yahoo Finance (last 5 years)
  - Property data parsed from Item 2 of 10-K/10-Q HTML filings

Usage:
    uv run python scripts/ingest_poc.py

Reads DATABASE_URL and EDGAR_USER_AGENT from the environment (or .env).
Creates the database tables if they do not yet exist.
"""

import sys
from pathlib import Path

# Allow running directly from the repo root without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker

from reit_dashboard.config import get_database_url, get_edgar_user_agent, get_gemini_api_key
from reit_dashboard.data.edgar_client import EdgarClient
from reit_dashboard.data.ingestion import (
    ingest_all_poc_reits,
    ingest_all_property_data,
    ingest_all_stock_prices,
)
from reit_dashboard.data.models import Base
from reit_dashboard.data.stock_price_client import StockPriceClient


def _apply_schema_migrations(engine) -> None:
    """
    Apply incremental schema changes to existing databases.

    SQLAlchemy's create_all only creates missing *tables*; it does not
    add new columns to existing tables.  This function handles the column
    additions introduced after the initial schema version.

    Parameters:
        engine: Bound SQLAlchemy engine.
    """
    inspector = inspect(engine)
    existing_tables = inspector.get_table_names()

    if "companies" in existing_tables:
        existing_cols = {
            c["name"] for c in inspector.get_columns("companies")
        }
        if "ticker" not in existing_cols:
            with engine.connect() as conn:
                conn.execute(
                    text("ALTER TABLE companies ADD COLUMN ticker VARCHAR(20)")
                )
                conn.commit()
            print("  [migration] Added 'ticker' column to companies table.")


def main() -> None:
    """
    Initialise the database and run EDGAR + stock price + property ingestion.

    Prints a per-company summary table to stdout.
    """
    db_url = get_database_url()
    user_agent = get_edgar_user_agent()

    print(f"Connecting to database: {db_url}")
    engine = create_engine(db_url, echo=False)

    # Apply any pending schema migrations before create_all.
    _apply_schema_migrations(engine)

    # Create all tables (idempotent – safe to run multiple times).
    Base.metadata.create_all(engine)

    SessionFactory = sessionmaker(bind=engine)
    edgar_client = EdgarClient(user_agent=user_agent)
    stock_client = StockPriceClient()

    # ------------------------------------------------------------------ #
    # 1. EDGAR: company metadata + quarterly financial facts              #
    # ------------------------------------------------------------------ #
    with SessionFactory() as session:
        print("\n=== EDGAR ingestion (10-Q, last 5 years) ===\n")
        edgar_results = ingest_all_poc_reits(edgar_client, session)

    print(f"{'Company':<30} {'Facts':>8} {'Concepts':>10}  Status")
    print("-" * 62)
    for r in edgar_results:
        status = f"ERROR: {r['error']}" if r["error"] else "OK"
        print(
            f"{r['name']:<30} {r['facts_upserted']:>8} "
            f"{r['concepts_fetched']:>10}  {status}"
        )

    # ------------------------------------------------------------------ #
    # 2. Yahoo Finance: weekly stock prices (last 5 years)               #
    # ------------------------------------------------------------------ #
    with SessionFactory() as session:
        print("\n=== Stock price ingestion (weekly, last 5 years) ===\n")
        stock_results = ingest_all_stock_prices(stock_client, session)

    print(f"{'Company':<30} {'Ticker':>8} {'Prices':>8}  Status")
    print("-" * 62)
    for r in stock_results:
        status = f"ERROR: {r['error']}" if r["error"] else "OK"
        print(
            f"{r['name']:<30} {str(r.get('ticker', '')):>8} "
            f"{r['prices_upserted']:>8}  {status}"
        )

    # ------------------------------------------------------------------ #
    # 3. EDGAR HTML: Item 2 property tables + Gemini property counts      #
    # ------------------------------------------------------------------ #
    gemini_extractor = None
    gemini_key = get_gemini_api_key()
    if gemini_key:
        from reit_dashboard.data.gemini_client import GeminiPropertyExtractor
        gemini_extractor = GeminiPropertyExtractor(api_key=gemini_key)
        print("\n=== Property data ingestion (Item 2 HTML + Gemini) ===\n")
    else:
        print(
            "\n=== Property data ingestion (Item 2 HTML, no Gemini key set) ===\n"
            "  Set GEMINI_API_KEY in .env to enable Gemini property count extraction.\n"
        )

    with SessionFactory() as session:
        property_results = ingest_all_property_data(
            edgar_client, session, gemini_extractor=gemini_extractor
        )

    print(f"{'Company':<30} {'Filings':>8} {'Parsed':>8} {'Rows':>8} {'Gemini':>8}  Status")
    print("-" * 78)
    for r in property_results:
        status = f"ERROR: {r['error']}" if r["error"] else "OK"
        print(
            f"{r['name']:<30} {r['filings_processed']:>8} "
            f"{r['filings_parsed']:>8} {r['rows_upserted']:>8} "
            f"{r.get('gemini_counts', 0):>8}  {status}"
        )

    all_results = edgar_results + stock_results + property_results
    errors = [r for r in all_results if r["error"]]
    if errors:
        print(f"\n{len(errors)} step(s) failed.")
        sys.exit(1)
    else:
        print("\nIngestion complete.")


if __name__ == "__main__":
    main()
