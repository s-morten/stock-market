"""
Unit tests for reit_dashboard.data.edgar_client.

HTTP calls are intercepted by respx so no real network requests are made.
time.sleep is patched to keep tests fast.
"""

import json
import re
from datetime import date
from unittest.mock import patch

import httpx
import pytest
import respx

from reit_dashboard.data.edgar_client import (
    CompanyInfo,
    ConceptFacts,
    EdgarClient,
    POC_CONCEPTS,
)

USER_AGENT = "TestSuite test@example.com"


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    """
    Patch time.sleep globally so rate-limit delays don't slow down tests.
    """
    monkeypatch.setattr("reit_dashboard.data.edgar_client.time.sleep", lambda _: None)


# ---------------------------------------------------------------------------
# Fixtures – minimal EDGAR JSON payloads
# ---------------------------------------------------------------------------

SUBMISSIONS_PAYLOAD = {
    "name": "Prologis, Inc.",
    "sic": "6798",
    "fiscalYearEnd": "12-31",
}

CONCEPT_PAYLOAD = {
    "units": {
        "USD": [
            {
                "end": "2022-12-31",
                "val": 5_000_000,
                "form": "10-K",
                "accn": "0001045609-23-000001",
            },
            {
                "end": "2021-12-31",
                "val": 4_500_000,
                "form": "10-K",
                "accn": "0001045609-22-000001",
            },
            # Duplicate accession – should be de-duplicated.
            {
                "end": "2022-12-31",
                "val": 5_100_000,
                "form": "10-K",
                "accn": "0001045609-23-000001",
            },
            # Non-annual form – should be included (10-Q).
            {
                "end": "2022-09-30",
                "val": 1_200_000,
                "form": "10-Q",
                "accn": "0001045609-22-000099",
            },
            # Unsupported form – should be filtered out.
            {
                "end": "2022-12-31",
                "val": 5_200_000,
                "form": "8-K",
                "accn": "0001045609-23-000002",
            },
        ]
    }
}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPadCik:
    """Tests for the static _pad_cik helper."""

    def test_already_padded(self):
        """A 10-digit CIK should be returned unchanged."""
        assert EdgarClient._pad_cik("0001045609") == "0001045609"

    def test_short_cik_is_padded(self):
        """A short CIK should be zero-padded to 10 digits."""
        assert EdgarClient._pad_cik("1045609") == "0001045609"

    def test_leading_zeros_stripped_then_padded(self):
        """Leading zeros are stripped before re-padding."""
        assert EdgarClient._pad_cik("00000001") == "0000000001"


class TestFetchCompanyInfo:
    """Tests for EdgarClient.fetch_company_info."""

    @respx.mock
    def test_happy_path(self):
        """Successful response is parsed into CompanyInfo."""
        respx.get(
            "https://data.sec.gov/submissions/CIK0001045609.json"
        ).mock(return_value=httpx.Response(200, json=SUBMISSIONS_PAYLOAD))

        client = EdgarClient(user_agent=USER_AGENT)
        info = client.fetch_company_info("0001045609")

        assert isinstance(info, CompanyInfo)
        assert info.name == "Prologis, Inc."
        assert info.sic == "6798"
        assert info.fiscal_year_end == "12-31"

    @respx.mock
    def test_404_raises(self):
        """A 404 response should raise httpx.HTTPStatusError."""
        respx.get(
            "https://data.sec.gov/submissions/CIK0001045609.json"
        ).mock(return_value=httpx.Response(404))

        client = EdgarClient(user_agent=USER_AGENT)
        with pytest.raises(httpx.HTTPStatusError):
            client.fetch_company_info("0001045609")

    @respx.mock
    def test_user_agent_header_sent(self):
        """The configured User-Agent must be present in outgoing requests."""
        route = respx.get(
            "https://data.sec.gov/submissions/CIK0001045609.json"
        ).mock(return_value=httpx.Response(200, json=SUBMISSIONS_PAYLOAD))

        client = EdgarClient(user_agent=USER_AGENT)
        client.fetch_company_info("0001045609")

        assert route.calls[0].request.headers["user-agent"] == USER_AGENT


class TestFetchConceptFacts:
    """Tests for EdgarClient.fetch_concept_facts."""

    @respx.mock
    def test_returns_concept_facts(self):
        """Valid payload returns a ConceptFacts with de-duplicated entries."""
        respx.get(
            "https://data.sec.gov/api/xbrl/companyconcept/"
            "CIK0001045609/us-gaap/Revenues.json"
        ).mock(return_value=httpx.Response(200, json=CONCEPT_PAYLOAD))

        client = EdgarClient(user_agent=USER_AGENT)
        result = client.fetch_concept_facts("0001045609", "Revenues")

        assert isinstance(result, ConceptFacts)
        assert result.concept == "Revenues"
        # De-duplication: one entry per unique accession → 3 entries remain.
        assert len(result.entries) == 3

    @respx.mock
    def test_8k_forms_excluded(self):
        """8-K filings must not appear in the returned entries."""
        respx.get(
            "https://data.sec.gov/api/xbrl/companyconcept/"
            "CIK0001045609/us-gaap/Revenues.json"
        ).mock(return_value=httpx.Response(200, json=CONCEPT_PAYLOAD))

        client = EdgarClient(user_agent=USER_AGENT)
        result = client.fetch_concept_facts("0001045609", "Revenues")

        forms = {e.form for e in result.entries}
        assert "8-K" not in forms

    @respx.mock
    def test_fetch_all_poc_facts_skips_missing(self):
        """A 404 on a concept should be silently skipped."""
        # Only Revenues returns 200; all others return 404.
        base = (
            "https://data.sec.gov/api/xbrl/companyconcept/"
            "CIK0001045609/us-gaap/"
        )
        respx.get(base + "Revenues.json").mock(
            return_value=httpx.Response(200, json=CONCEPT_PAYLOAD)
        )
        for concept in POC_CONCEPTS[1:]:
            respx.get(base + f"{concept}.json").mock(
                return_value=httpx.Response(404)
            )

        client = EdgarClient(user_agent=USER_AGENT)
        results = client.fetch_all_poc_facts("0001045609")

        assert len(results) == 1
        assert results[0].concept == "Revenues"


class TestSinceDateFilter:
    """Tests for the since_date parameter in fetch_concept_facts."""

    @respx.mock
    def test_since_date_excludes_old_entries(self):
        """Entries before since_date must not appear in results."""
        respx.get(
            "https://data.sec.gov/api/xbrl/companyconcept/"
            "CIK0001045609/us-gaap/Revenues.json"
        ).mock(return_value=httpx.Response(200, json=CONCEPT_PAYLOAD))

        client = EdgarClient(user_agent=USER_AGENT)
        # Cut-off after the 2021-12-31 entry; only 2022 entries should remain.
        result = client.fetch_concept_facts(
            "0001045609", "Revenues", since_date=date(2022, 1, 1)
        )

        for entry in result.entries:
            assert entry.end >= date(2022, 1, 1)

    @respx.mock
    def test_since_date_none_keeps_all(self):
        """When since_date is None all passing entries are returned."""
        respx.get(
            "https://data.sec.gov/api/xbrl/companyconcept/"
            "CIK0001045609/us-gaap/Revenues.json"
        ).mock(return_value=httpx.Response(200, json=CONCEPT_PAYLOAD))

        client = EdgarClient(user_agent=USER_AGENT)
        result = client.fetch_concept_facts(
            "0001045609", "Revenues", since_date=None
        )
        # 3 unique accessions (8-K filtered, duplicate merged).
        assert len(result.entries) == 3


class TestFormsFilter:
    """Tests for the forms parameter in fetch_concept_facts."""

    @respx.mock
    def test_only_10q_when_forms_restricted(self):
        """Passing forms={'10-Q'} should exclude all 10-K entries."""
        respx.get(
            "https://data.sec.gov/api/xbrl/companyconcept/"
            "CIK0001045609/us-gaap/Revenues.json"
        ).mock(return_value=httpx.Response(200, json=CONCEPT_PAYLOAD))

        client = EdgarClient(user_agent=USER_AGENT)
        result = client.fetch_concept_facts(
            "0001045609", "Revenues", forms={"10-Q"}
        )

        assert all(e.form == "10-Q" for e in result.entries)

    @respx.mock
    def test_rate_limit_sleep_called(self, monkeypatch):
        """time.sleep should be called once per HTTP request."""
        sleep_calls: list[float] = []
        monkeypatch.setattr(
            "reit_dashboard.data.edgar_client.time.sleep",
            lambda s: sleep_calls.append(s),
        )
        respx.get(
            "https://data.sec.gov/submissions/CIK0001045609.json"
        ).mock(return_value=httpx.Response(200, json=SUBMISSIONS_PAYLOAD))

        client = EdgarClient(user_agent=USER_AGENT)
        client.fetch_company_info("0001045609")

        assert len(sleep_calls) == 1
        assert sleep_calls[0] > 0


# ---------------------------------------------------------------------------
# Tests for non-monetary property concepts (unit=None auto-detect)
# ---------------------------------------------------------------------------

PROPERTY_COUNT_PAYLOAD = {
    "units": {
        "Property": [
            {
                "end": "2023-12-31",
                "val": 2500,
                "form": "10-K",
                "accn": "0001045609-24-000001",
            },
            {
                "end": "2022-12-31",
                "val": 2300,
                "form": "10-K",
                "accn": "0001045609-23-000001",
            },
        ]
    }
}

MIXED_UNITS_PAYLOAD = {
    "units": {
        "Property": [
            {"end": "2023-12-31", "val": 10, "form": "10-K", "accn": "A1"},
        ],
        "properties": [
            {"end": "2023-12-31", "val": 100, "form": "10-K", "accn": "A2"},
            {"end": "2022-12-31", "val": 90, "form": "10-K", "accn": "A3"},
        ],
    }
}


class TestUnitAutoDetect:
    """Tests for unit=None auto-detection in fetch_concept_facts."""

    @respx.mock
    def test_auto_detect_unit_picks_largest(self):
        """unit=None should pick the unit with the most entries."""
        respx.get(
            "https://data.sec.gov/api/xbrl/companyconcept/"
            "CIK0001045609/us-gaap/SomeConcept.json"
        ).mock(return_value=httpx.Response(200, json=MIXED_UNITS_PAYLOAD))

        client = EdgarClient(user_agent=USER_AGENT)
        result = client.fetch_concept_facts("0001045609", "SomeConcept", unit=None)

        # "properties" has 2 entries vs "Property" with 1 → auto-picks "properties"
        assert result.unit == "properties"
        assert len(result.entries) == 2

    @respx.mock
    def test_fetch_concept_facts_with_custom_unit(self):
        """Non-USD unit facts are returned correctly when unit=None."""
        respx.get(
            "https://data.sec.gov/api/xbrl/companyconcept/"
            "CIK0001045609/us-gaap/SomeConcept.json"
        ).mock(return_value=httpx.Response(200, json=PROPERTY_COUNT_PAYLOAD))

        client = EdgarClient(user_agent=USER_AGENT)
        result = client.fetch_concept_facts("0001045609", "SomeConcept", unit=None)

        assert result.unit == "Property"
        assert len(result.entries) == 2

    @respx.mock
    def test_no_units_in_response_raises_key_error(self):
        """Empty units dict with unit=None should raise KeyError."""
        respx.get(
            "https://data.sec.gov/api/xbrl/companyconcept/"
            "CIK0001045609/us-gaap/SomeConcept.json"
        ).mock(return_value=httpx.Response(200, json={"units": {}}))

        client = EdgarClient(user_agent=USER_AGENT)
        with pytest.raises(KeyError):
            client.fetch_concept_facts("0001045609", "SomeConcept", unit=None)


# ---------------------------------------------------------------------------
# Tests for GeminiPropertyExtractor
# ---------------------------------------------------------------------------


class TestGeminiPropertyExtractor:
    """Unit tests for GeminiPropertyExtractor (HTTP calls mocked)."""

    GEMINI_URL_PATTERN = re.compile(
        r"https://generativelanguage\.googleapis\.com/.*generateContent.*"
    )

    @staticmethod
    def _gemini_response(total_properties, notes="ok"):
        body = json.dumps({"total_properties": total_properties, "notes": notes})
        return httpx.Response(200, json={
            "candidates": [{"content": {"parts": [{"text": body}]}}]
        })

    @respx.mock
    def test_returns_property_count(self, monkeypatch):
        """A valid Gemini response is parsed to a dict with total_properties."""
        from reit_dashboard.data.gemini_client import GeminiPropertyExtractor

        monkeypatch.setattr("reit_dashboard.data.gemini_client.time.sleep", lambda _: None)
        respx.post(self.GEMINI_URL_PATTERN).mock(
            return_value=self._gemini_response(1500, "Found total row = 1500")
        )

        extractor = GeminiPropertyExtractor(api_key="test-key")
        result = extractor.extract_property_count("Property Type\tCount\nOffice\t500\nTotal\t1500")

        assert result["total_properties"] == 1500
        assert "notes" in result

    @respx.mock
    def test_null_count_when_not_found(self, monkeypatch):
        """Gemini returning null total_properties is propagated correctly."""
        from reit_dashboard.data.gemini_client import GeminiPropertyExtractor

        monkeypatch.setattr("reit_dashboard.data.gemini_client.time.sleep", lambda _: None)
        respx.post(self.GEMINI_URL_PATTERN).mock(
            return_value=self._gemini_response(None, "Cannot determine")
        )

        extractor = GeminiPropertyExtractor(api_key="test-key")
        result = extractor.extract_property_count("some table text")

        assert result["total_properties"] is None

    def test_empty_tables_text_skips_api(self, monkeypatch):
        """Empty tables_text must not call the Gemini API."""
        from reit_dashboard.data.gemini_client import GeminiPropertyExtractor

        called = []
        monkeypatch.setattr("reit_dashboard.data.gemini_client.time.sleep", lambda _: None)

        extractor = GeminiPropertyExtractor(api_key="test-key")
        # Patch _client to track calls
        original_post = httpx.Client.post

        def track_post(self, *a, **kw):
            called.append(1)
            return original_post(self, *a, **kw)

        result = extractor.extract_property_count("")

        assert not called
        assert result["total_properties"] is None

