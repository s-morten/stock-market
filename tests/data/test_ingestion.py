"""
Unit tests for reit_dashboard.data.ingestion.

The EdgarClient is mocked so no network calls are made.
Tests verify the orchestration logic (what gets persisted, how errors
are handled) rather than the HTTP or SQL details.
"""

from datetime import date
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from reit_dashboard.data.edgar_client import (
    CompanyInfo,
    ConceptFacts,
    FactEntry,
)
from reit_dashboard.data.ingestion import POC_REITS, ingest_all_poc_reits, ingest_company
from reit_dashboard.data.repository import CompanyRepository, FinancialFactRepository


def _make_client_mock(
    cik: str = "0001045609",
    name: str = "Prologis, Inc.",
    concepts: list[str] | None = None,
) -> MagicMock:
    """
    Build a MagicMock that behaves like an EdgarClient.

    Parameters:
        cik:      CIK returned by fetch_company_info.
        name:     Company name returned by fetch_company_info.
        concepts: Concepts to return facts for (default: ["Revenues"]).

    Returns:
        MagicMock: Pre-configured client mock.
    """
    if concepts is None:
        concepts = ["Revenues"]

    client = MagicMock()
    client.fetch_company_info.return_value = CompanyInfo(
        cik=cik, name=name, sic="6798", fiscalYearEnd="12-31"
    )
    client.fetch_all_poc_facts.return_value = [
        ConceptFacts(
            concept=c,
            unit="USD",
            entries=[
                FactEntry(
                    end=date(2022, 12, 31),
                    val=5_000_000.0,
                    form="10-K",
                    accn=f"accn-{c}-2022",
                ),
            ],
        )
        for c in concepts
    ]
    return client


class TestIngestCompany:
    """Tests for the ingest_company function."""

    def test_persists_company_and_facts(self, db_session):
        """Successful ingestion stores company metadata and facts."""
        client = _make_client_mock(concepts=["Revenues", "Assets"])
        result = ingest_company("0001045609", client, db_session)
        db_session.commit()

        company_repo = CompanyRepository(db_session)
        fact_repo = FinancialFactRepository(db_session)

        assert company_repo.get_by_cik("0001045609") is not None
        facts = fact_repo.get_facts("0001045609")
        assert len(facts) == 2
        assert result["facts_upserted"] == 2
        assert result["concepts_fetched"] == 2

    def test_returns_correct_summary(self, db_session):
        """Return dict contains the expected keys and values."""
        client = _make_client_mock(concepts=["Revenues"])
        result = ingest_company("0001045609", client, db_session)

        assert "facts_upserted" in result
        assert "concepts_fetched" in result
        assert result["concepts_fetched"] == 1

    def test_calls_fetch_company_info(self, db_session):
        """fetch_company_info is called exactly once with the CIK."""
        client = _make_client_mock()
        ingest_company("0001045609", client, db_session)
        client.fetch_company_info.assert_called_once_with("0001045609")


class TestIngestAllPocReits:
    """Tests for the ingest_all_poc_reits function."""

    def test_processes_all_poc_reits(self, db_session):
        """All companies in POC_REITS are attempted."""
        client = _make_client_mock()
        # Override return to handle any CIK.
        client.fetch_company_info.side_effect = lambda cik: CompanyInfo(
            cik=cik, name=f"Company {cik}", sic="6798", fiscalYearEnd="12-31"
        )
        client.fetch_all_poc_facts.return_value = []

        results = ingest_all_poc_reits(client, db_session)

        assert len(results) == len(POC_REITS)

    def test_error_in_one_does_not_stop_others(self, db_session):
        """A failure for one company is captured; others continue."""
        call_count = 0
        expected = len(POC_REITS)

        def side_effect(cik):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("Simulated EDGAR error")
            return CompanyInfo(
                cik=cik, name=f"Company {cik}", sic="6798", fiscalYearEnd="12-31"
            )

        client = MagicMock()
        client.fetch_company_info.side_effect = side_effect
        client.fetch_all_poc_facts.return_value = []

        results = ingest_all_poc_reits(client, db_session)

        assert len(results) == expected
        errors = [r for r in results if r["error"]]
        successes = [r for r in results if not r["error"]]
        assert len(errors) == 1
        assert len(successes) == expected - 1
