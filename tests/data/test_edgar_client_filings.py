"""
Unit tests for EdgarClient CIK lookup and filing-list methods.
"""

from datetime import date
from unittest.mock import patch

import httpx
import pytest
import respx

from reit_dashboard.data.edgar_client import EdgarClient

USER_AGENT = "TestSuite test@example.com"


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr("reit_dashboard.data.edgar_client.time.sleep", lambda _: None)


_TICKERS_PAYLOAD = {
    "0": {"cik_str": 1045609, "ticker": "PLD", "title": "Prologis Inc"},
    "1": {"cik_str": 726854, "ticker": "O", "title": "Realty Income Corp"},
}

_SUBMISSIONS_PAYLOAD = {
    "name": "Prologis, Inc.",
    "sic": "6798",
    "fiscalYearEnd": "12-31",
    "filings": {
        "recent": {
            "accessionNumber": [
                "0001045609-24-000001",
                "0001045609-23-000099",
                "0001045609-23-000055",
            ],
            "form": ["10-K", "10-Q", "8-K"],
            "filingDate": ["2024-02-15", "2023-10-25", "2023-08-01"],
            "reportDate": ["2023-12-31", "2023-09-30", ""],
            "primaryDocument": [
                "pld-20231231.htm",
                "pld-20230930.htm",
                "pld-8k.htm",
            ],
        }
    },
}

_INDEX_PAYLOAD = {
    "documents": [
        {"type": "10-K", "name": "pld-20231231.htm"},
        {"type": "XML", "name": "pld-20231231_htm.xml"},
    ]
}


class TestGetCikForTicker:
    @respx.mock
    def test_found(self):
        respx.get("https://www.sec.gov/files/company_tickers.json").mock(
            return_value=httpx.Response(200, json=_TICKERS_PAYLOAD)
        )
        client = EdgarClient(user_agent=USER_AGENT)
        cik = client.get_cik_for_ticker("PLD")
        assert cik == "0001045609"

    @respx.mock
    def test_case_insensitive(self):
        respx.get("https://www.sec.gov/files/company_tickers.json").mock(
            return_value=httpx.Response(200, json=_TICKERS_PAYLOAD)
        )
        client = EdgarClient(user_agent=USER_AGENT)
        assert client.get_cik_for_ticker("pld") == "0001045609"

    @respx.mock
    def test_not_found_returns_none(self):
        respx.get("https://www.sec.gov/files/company_tickers.json").mock(
            return_value=httpx.Response(200, json=_TICKERS_PAYLOAD)
        )
        client = EdgarClient(user_agent=USER_AGENT)
        assert client.get_cik_for_ticker("UNKNWN") is None

    @respx.mock
    def test_zero_pads_cik(self):
        respx.get("https://www.sec.gov/files/company_tickers.json").mock(
            return_value=httpx.Response(200, json=_TICKERS_PAYLOAD)
        )
        client = EdgarClient(user_agent=USER_AGENT)
        cik = client.get_cik_for_ticker("O")
        assert cik == "0000726854"


class TestListFilings:
    @respx.mock
    def test_returns_only_requested_forms(self):
        respx.get(
            "https://data.sec.gov/submissions/CIK0001045609.json"
        ).mock(return_value=httpx.Response(200, json=_SUBMISSIONS_PAYLOAD))

        client = EdgarClient(user_agent=USER_AGENT)
        filings = client.list_filings("0001045609", forms={"10-K", "10-Q"})

        forms = {f["form"] for f in filings}
        assert "8-K" not in forms
        assert len(filings) == 2

    @respx.mock
    def test_since_date_filter(self):
        respx.get(
            "https://data.sec.gov/submissions/CIK0001045609.json"
        ).mock(return_value=httpx.Response(200, json=_SUBMISSIONS_PAYLOAD))

        client = EdgarClient(user_agent=USER_AGENT)
        filings = client.list_filings(
            "0001045609",
            forms={"10-K", "10-Q"},
            since_date=date(2024, 1, 1),
        )
        # Only the 2024-02-15 10-K qualifies.
        assert len(filings) == 1
        assert filings[0]["form"] == "10-K"

    @respx.mock
    def test_max_filings_cap(self):
        respx.get(
            "https://data.sec.gov/submissions/CIK0001045609.json"
        ).mock(return_value=httpx.Response(200, json=_SUBMISSIONS_PAYLOAD))

        client = EdgarClient(user_agent=USER_AGENT)
        filings = client.list_filings(
            "0001045609",
            forms={"10-K", "10-Q"},
            max_filings=1,
        )
        assert len(filings) == 1

    @respx.mock
    def test_filing_keys(self):
        respx.get(
            "https://data.sec.gov/submissions/CIK0001045609.json"
        ).mock(return_value=httpx.Response(200, json=_SUBMISSIONS_PAYLOAD))

        client = EdgarClient(user_agent=USER_AGENT)
        filings = client.list_filings("0001045609")

        for f in filings:
            assert "accn" in f
            assert "form" in f
            assert "filingDate" in f
            assert "reportDate" in f
            assert "primaryDocument" in f

    @respx.mock
    def test_primary_document_included(self):
        """list_filings returns the primaryDocument field from submissions."""
        respx.get(
            "https://data.sec.gov/submissions/CIK0001045609.json"
        ).mock(return_value=httpx.Response(200, json=_SUBMISSIONS_PAYLOAD))

        client = EdgarClient(user_agent=USER_AGENT)
        filings = client.list_filings("0001045609", forms={"10-K"})

        assert filings[0]["primaryDocument"] == "pld-20231231.htm"


class TestFetchFilingHtml:
    @respx.mock
    def test_happy_path_with_primary_doc(self):
        """When primary_doc is supplied the document is fetched directly."""
        accn = "0001045609-24-000001"
        accn_nodash = "000104560924000001"
        cik_int = 1045609

        doc_url = (
            f"https://www.sec.gov/Archives/edgar/data/"
            f"{cik_int}/{accn_nodash}/pld-20231231.htm"
        )
        respx.get(doc_url).mock(
            return_value=httpx.Response(200, text="<html>filing content</html>")
        )

        client = EdgarClient(user_agent=USER_AGENT)
        html = client.fetch_filing_html("0001045609", accn, primary_doc="pld-20231231.htm")
        assert "filing content" in html

    @respx.mock
    def test_fallback_to_index_json(self):
        """Without primary_doc the method falls back to index.json discovery."""
        accn = "0001045609-24-000001"
        accn_nodash = "000104560924000001"
        cik_int = 1045609

        index_url = (
            f"https://www.sec.gov/Archives/edgar/data/"
            f"{cik_int}/{accn_nodash}/index.json"
        )
        doc_url = (
            f"https://www.sec.gov/Archives/edgar/data/"
            f"{cik_int}/{accn_nodash}/pld-20231231.htm"
        )
        respx.get(index_url).mock(
            return_value=httpx.Response(200, json=_INDEX_PAYLOAD)
        )
        respx.get(doc_url).mock(
            return_value=httpx.Response(200, text="<html>fallback content</html>")
        )

        client = EdgarClient(user_agent=USER_AGENT)
        html = client.fetch_filing_html("0001045609", accn)
        assert "fallback content" in html

    @respx.mock
    def test_no_primary_doc_raises(self):
        """ValueError when index.json has no matching HTML document."""
        accn = "0001045609-24-000001"
        accn_nodash = "000104560924000001"
        cik_int = 1045609

        index_url = (
            f"https://www.sec.gov/Archives/edgar/data/"
            f"{cik_int}/{accn_nodash}/index.json"
        )
        respx.get(index_url).mock(
            return_value=httpx.Response(
                200, json={"documents": [{"type": "XML", "name": "data.xml"}]}
            )
        )

        client = EdgarClient(user_agent=USER_AGENT)
        with pytest.raises(ValueError, match="No primary HTML document"):
            client.fetch_filing_html("0001045609", accn)
