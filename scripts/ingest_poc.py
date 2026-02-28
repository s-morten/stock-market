"""
PoC ingestion script – seeds the database with data for 5 US REITs.

Usage:
    python scripts/ingest_poc.py

Reads DATABASE_URL and EDGAR_USER_AGENT from the environment (or .env).
Creates the database tables if they do not yet exist.
"""

import sys
from pathlib import Path

# Allow running directly from the repo root without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from reit_dashboard.config import get_database_url, get_edgar_user_agent
from reit_dashboard.data.edgar_client import EdgarClient
from reit_dashboard.data.ingestion import ingest_all_poc_reits
from reit_dashboard.data.models import Base


def main() -> None:
    """
    Initialise the database and run ingestion for all PoC REITs.

    Prints a per-company summary table to stdout.
    """
    db_url = get_database_url()
    user_agent = get_edgar_user_agent()

    print(f"Connecting to database: {db_url}")
    engine = create_engine(db_url, echo=False)

    # Create all tables (idempotent – safe to run multiple times).
    Base.metadata.create_all(engine)

    SessionFactory = sessionmaker(bind=engine)
    client = EdgarClient(user_agent=user_agent)

    with SessionFactory() as session:
        print("\nStarting EDGAR ingestion for PoC REITs...\n")
        results = ingest_all_poc_reits(client, session)

    # Print summary table.
    print(f"{'Company':<30} {'Facts':>8} {'Concepts':>10}  Status")
    print("-" * 60)
    for r in results:
        status = f"ERROR: {r['error']}" if r["error"] else "OK"
        print(
            f"{r['name']:<30} {r['facts_upserted']:>8} "
            f"{r['concepts_fetched']:>10}  {status}"
        )

    errors = [r for r in results if r["error"]]
    if errors:
        print(f"\n{len(errors)} company/ies failed.")
        sys.exit(1)
    else:
        print("\nIngestion complete.")


if __name__ == "__main__":
    main()
