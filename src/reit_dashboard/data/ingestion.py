"""
Data ingestion service for REIT Dashboard.

Orchestrates the end-to-end pipeline:
  1. Fetch company metadata from EDGAR.
  2. Fetch XBRL financial facts for each PoC concept.
  3. Validate and transform via Pydantic models.
  4. Persist to the database via repository layer.

This module owns the business logic; it depends on EdgarClient and the
repository classes but does not know about HTTP or SQL details directly.
"""

from decimal import Decimal

from sqlalchemy.orm import Session

from reit_dashboard.data.edgar_client import EdgarClient
from reit_dashboard.data.models import Company, FinancialFact
from reit_dashboard.data.repository import (
    CompanyRepository,
    FinancialFactRepository,
)

# CIK → display-name mapping for the PoC REIT universe.
POC_REITS: dict[str, str] = {
    "0001045609": "Prologis",
    "0000726854": "Realty Income",
    "0001063761": "Simon Property Group",
    "0001393311": "Public Storage",
    "0000766704": "Welltower",
}


def ingest_company(
    cik: str,
    client: EdgarClient,
    session: Session,
) -> dict[str, int]:
    """
    Ingest one REIT company: metadata + all PoC financial facts.

    Parameters:
        cik:     Company CIK (raw or zero-padded).
        client:  Configured EdgarClient instance.
        session: Active SQLAlchemy session (not yet committed).

    Returns:
        dict: Summary with keys "facts_upserted" and "concepts_fetched".
    """
    company_repo = CompanyRepository(session)
    fact_repo = FinancialFactRepository(session)

    # --- 1. Company metadata ---
    info = client.fetch_company_info(cik)
    company = Company(
        cik=info.cik,
        name=info.name,
        sic=info.sic,
        fiscal_year_end=info.fiscal_year_end,
    )
    company_repo.upsert(company)

    # --- 2. Financial facts ---
    all_concept_facts = client.fetch_all_poc_facts(cik)
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
