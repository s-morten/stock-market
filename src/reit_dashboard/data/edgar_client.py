"""
EDGAR API client for fetching REIT company data from the SEC.

Uses the public SEC EDGAR REST API:
- Submissions endpoint: metadata about the company and its filings.
- Company Facts endpoint: XBRL financial data for all reported concepts.

The SEC requires a descriptive User-Agent header on every request.
All network calls use httpx for easy async upgrade and test mocking.
"""

import time
from datetime import date
from typing import Any

import httpx
from pydantic import BaseModel, Field

# Base URL for the EDGAR data API (not the search/EFTS endpoint).
_EDGAR_BASE = "https://data.sec.gov"

# GAAP concepts fetched for every company in the PoC.
POC_CONCEPTS = ["Revenues", "NetIncomeLoss", "Assets", "Liabilities"]

# Minimum seconds between outgoing HTTP requests.
# The SEC enforces a limit of 10 requests/second; 0.11 s gives a safe margin.
_REQUEST_DELAY_SECONDS = 0.11


# ---------------------------------------------------------------------------
# Pydantic schemas for EDGAR API responses
# ---------------------------------------------------------------------------


class CompanyInfo(BaseModel):
    """Parsed metadata from the EDGAR submissions endpoint."""

    cik: str
    name: str
    sic: str | None = None
    fiscal_year_end: str | None = Field(None, alias="fiscalYearEnd")

    model_config = {"populate_by_name": True}


class FactEntry(BaseModel):
    """
    A single data point returned inside the XBRL companyfacts response.

    Attributes:
        end:   Period end date (YYYY-MM-DD).
        val:   Reported numeric value.
        form:  Filing form type (e.g. "10-K").
        accn:  Accession number for the filing.
    """

    end: date
    val: float
    form: str
    accn: str


class ConceptFacts(BaseModel):
    """All fact entries for one XBRL concept and unit combination."""

    concept: str
    unit: str
    entries: list[FactEntry]


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class EdgarClient:
    """
    Thin HTTP wrapper around the SEC EDGAR public API.

    Parameters:
        user_agent: Required User-Agent string per SEC policy.
        timeout:    Request timeout in seconds (default 30).
    """

    def __init__(self, user_agent: str, timeout: float = 30.0) -> None:
        self._headers = {"User-Agent": user_agent}
        self._timeout = timeout

    def _get(self, url: str) -> dict[str, Any]:
        """
        Perform a GET request and return the parsed JSON body.

        A fixed delay of ``_REQUEST_DELAY_SECONDS`` is observed *before*
        every request so the caller never exceeds the SEC rate limit of
        10 requests per second, even when many concepts are fetched in a
        tight loop.

        Parameters:
            url: Absolute URL to request.

        Returns:
            dict: Parsed JSON response.

        Raises:
            httpx.HTTPStatusError: On 4xx/5xx responses.
        """
        time.sleep(_REQUEST_DELAY_SECONDS)
        with httpx.Client(headers=self._headers, timeout=self._timeout) as client:
            response = client.get(url)
            response.raise_for_status()
            return response.json()

    @staticmethod
    def _pad_cik(cik: str) -> str:
        """
        Zero-pad a CIK to 10 digits as required by the EDGAR API.

        Parameters:
            cik: Raw CIK string (e.g. "1045609" or "0001045609").

        Returns:
            str: 10-digit zero-padded CIK.
        """
        return cik.lstrip("0").zfill(10)

    def fetch_company_info(self, cik: str) -> CompanyInfo:
        """
        Fetch and parse company metadata from the submissions endpoint.

        Parameters:
            cik: Company CIK (raw or zero-padded).

        Returns:
            CompanyInfo: Parsed company metadata.
        """
        padded = self._pad_cik(cik)
        url = f"{_EDGAR_BASE}/submissions/CIK{padded}.json"
        data = self._get(url)
        return CompanyInfo(
            cik=padded,
            name=data["name"],
            sic=data.get("sic"),
            fiscalYearEnd=data.get("fiscalYearEnd"),
        )

    def fetch_concept_facts(
        self,
        cik: str,
        concept: str,
        taxonomy: str = "us-gaap",
        unit: str = "USD",
        since_date: date | None = None,
        forms: set[str] | None = None,
    ) -> ConceptFacts:
        """
        Fetch all reported values for a single XBRL concept.

        Only filings matching ``forms`` are included (default: 10-K and
        10-Q).  Duplicate accession numbers are de-duplicated, keeping
        the entry with the most recent period end date.  Entries with a
        ``period_end`` before ``since_date`` are discarded.

        Parameters:
            cik:        Company CIK.
            concept:    GAAP XBRL tag (e.g. "Revenues").
            taxonomy:   XBRL taxonomy namespace (default "us-gaap").
            unit:       Unit of measure to extract (default "USD").
            since_date: Exclude entries whose period end is before this
                        date.  Pass ``None`` to include all history.
            forms:      Set of filing form types to keep.  Defaults to
                        ``{"10-K", "10-Q"}``.

        Returns:
            ConceptFacts: Parsed fact entries for the concept.

        Raises:
            KeyError: If the concept or unit is not present in the data.
        """
        if forms is None:
            forms = {"10-K", "10-Q"}

        padded = self._pad_cik(cik)
        url = (
            f"{_EDGAR_BASE}/api/xbrl/companyconcept/"
            f"CIK{padded}/{taxonomy}/{concept}.json"
        )
        data = self._get(url)

        raw_entries: list[dict[str, Any]] = (
            data.get("units", {}).get(unit, [])
        )

        # Keep only requested form types; require a period end date.
        filtered = [
            e for e in raw_entries
            if e.get("form") in forms and "end" in e
        ]

        # Apply optional date cut-off.
        if since_date is not None:
            cutoff = since_date.isoformat()
            filtered = [e for e in filtered if e["end"] >= cutoff]

        # De-duplicate by accession number; keep the latest period end.
        seen: dict[str, dict[str, Any]] = {}
        for entry in filtered:
            accn = entry["accn"]
            if accn not in seen or entry["end"] > seen[accn]["end"]:
                seen[accn] = entry

        entries = [FactEntry(**e) for e in seen.values()]
        return ConceptFacts(concept=concept, unit=unit, entries=entries)

    def fetch_all_poc_facts(
        self,
        cik: str,
        since_date: date | None = None,
        forms: set[str] | None = None,
    ) -> list[ConceptFacts]:
        """
        Fetch all PoC XBRL concepts for a company.

        Silently skips any concept that is not reported by the company
        (e.g. some REITs use non-standard tags for revenue).

        Parameters:
            cik:        Company CIK.
            since_date: Earliest period end date to include (passed
                        through to :meth:`fetch_concept_facts`).
            forms:      Filing form types to include (passed through to
                        :meth:`fetch_concept_facts`).

        Returns:
            list[ConceptFacts]: One entry per successfully fetched concept.
        """
        results: list[ConceptFacts] = []
        for concept in POC_CONCEPTS:
            try:
                results.append(
                    self.fetch_concept_facts(
                        cik,
                        concept,
                        since_date=since_date,
                        forms=forms,
                    )
                )
            except (httpx.HTTPStatusError, KeyError):
                # Concept not available for this company – skip gracefully.
                pass
        return results
