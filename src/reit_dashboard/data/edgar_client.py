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

# GAAP concepts fetched for every company in the PoC (all denominated in USD).
POC_CONCEPTS = ["Revenues", "NetIncomeLoss", "Assets", "Liabilities"]

# Non-monetary property-portfolio concepts; unit is auto-detected from the
# response because different companies use different unit labels.
POC_PROPERTY_CONCEPTS = [
    "NumberOfRealEstateProperties",  # unit: "Property", "properties", "item", …
    "AreaOfRealEstateProperty",      # unit: "sqft"
]

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
        unit: str | None = "USD",
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
                        Pass ``None`` to auto-select the first available
                        unit – useful for non-monetary concepts such as
                        ``NumberOfRealEstateProperties`` where the unit
                        label varies by company.
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

        units_map: dict[str, list[dict[str, Any]]] = data.get("units", {})

        if unit is None:
            # Auto-detect: pick the unit with the most facts entries.
            if not units_map:
                raise KeyError(f"No units found for concept {concept}")
            unit = max(units_map, key=lambda k: len(units_map[k]))

        raw_entries: list[dict[str, Any]] = units_map.get(unit, [])

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
        # Property-portfolio concepts use non-monetary units; auto-detect.
        for concept in POC_PROPERTY_CONCEPTS:
            try:
                results.append(
                    self.fetch_concept_facts(
                        cik,
                        concept,
                        unit=None,  # auto-detect unit label
                        since_date=since_date,
                        forms=forms,
                    )
                )
            except (httpx.HTTPStatusError, KeyError):
                pass
        return results

    # ------------------------------------------------------------------
    # CIK lookup
    # ------------------------------------------------------------------

    def get_cik_for_ticker(self, ticker: str) -> str | None:
        """
        Resolve a stock ticker to a SEC CIK using the SEC's public
        company-tickers JSON file.

        The mapping is fetched once per client call (no local cache here;
        callers may cache the result themselves).

        Parameters:
            ticker: Exchange ticker symbol, case-insensitive (e.g. "PLD").

        Returns:
            str: Zero-padded 10-digit CIK, or ``None`` if not found.
        """
        url = "https://www.sec.gov/files/company_tickers.json"
        data = self._get(url)
        target = ticker.upper()
        for entry in data.values():
            if entry.get("ticker", "").upper() == target:
                cik_int: int = entry["cik_str"]
                return str(cik_int).zfill(10)
        return None

    # ------------------------------------------------------------------
    # Filing index
    # ------------------------------------------------------------------

    def list_filings(
        self,
        cik: str,
        forms: set[str] | None = None,
        since_date: date | None = None,
        max_filings: int | None = None,
    ) -> list[dict[str, str]]:
        """
        List filings for a company from the EDGAR submissions endpoint.

        Returns a list of dicts with keys:
        ``accn``, ``form``, ``filingDate``, ``reportDate``,
        ``primaryDocument``.

        The ``primaryDocument`` field is the filename of the primary
        filing document as reported by SEC EDGAR (e.g. "pld-20231231.htm").
        It can be passed directly to :meth:`fetch_filing_html` to avoid a
        separate index-page request.

        Filings are returned in reverse-chronological order (newest first).

        Parameters:
            cik:          Company CIK (raw or zero-padded).
            forms:        Filing form types to include (e.g. {"10-K", "10-Q"}).
                          Defaults to both.
            since_date:   Exclude filings before this date.
            max_filings:  Cap the number of results.

        Returns:
            list[dict]: Filing metadata records.
        """
        if forms is None:
            forms = {"10-K", "10-Q"}

        padded = self._pad_cik(cik)
        url = f"{_EDGAR_BASE}/submissions/CIK{padded}.json"
        data = self._get(url)

        recent = data.get("filings", {}).get("recent", {})
        accns = recent.get("accessionNumber", [])
        form_list = recent.get("form", [])
        dates = recent.get("filingDate", [])
        periods = recent.get("reportDate", [])
        primary_docs = recent.get("primaryDocument", [""] * len(accns))

        results: list[dict[str, str]] = []
        for accn, form, filing_date, report_date, primary_doc in zip(
            accns, form_list, dates, periods, primary_docs
        ):
            if form not in forms:
                continue
            if since_date is not None and filing_date < since_date.isoformat():
                continue
            results.append(
                {
                    "accn": accn,
                    "form": form,
                    "filingDate": filing_date,
                    "reportDate": report_date,
                    "primaryDocument": primary_doc or "",
                }
            )
            if max_filings is not None and len(results) >= max_filings:
                break

        return results

    def fetch_filing_html(
        self,
        cik: str,
        accn: str,
        primary_doc: str | None = None,
    ) -> str:
        """
        Fetch the primary HTML document for a given SEC filing.

        When ``primary_doc`` is provided (the filename as returned by
        :meth:`list_filings`) the document is fetched directly with a
        single HTTP request.  When it is ``None`` the method falls back to
        fetching the filing's ``index.json`` to discover the primary
        document name.

        Parameters:
            cik:         Company CIK (raw or zero-padded).
            accn:        Accession number (with or without dashes).
            primary_doc: Optional filename of the primary document
                         (e.g. "pld-20231231.htm").

        Returns:
            str: Raw HTML text of the primary filing document.

        Raises:
            httpx.HTTPStatusError: On HTTP errors.
            ValueError: If no primary HTML document can be located.
        """
        padded = self._pad_cik(cik)
        # Accession number in the URL uses no dashes.
        accn_nodash = accn.replace("-", "")
        cik_int = int(padded)
        base = f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{accn_nodash}"

        if not primary_doc:
            # Fall back: fetch the filing index JSON to discover the filename.
            index_url = f"{base}/index.json"
            time.sleep(_REQUEST_DELAY_SECONDS)
            with httpx.Client(headers=self._headers, timeout=self._timeout) as client:
                resp = client.get(index_url)
                resp.raise_for_status()
                index = resp.json()

            # Find the primary document (.htm whose type matches the form).
            _FORM_TYPES = {"10-K", "10-Q", "10-K/A", "10-Q/A"}
            primary_doc = None
            for item in index.get("documents", []):
                if item.get("type") in _FORM_TYPES:
                    name: str = item.get("name", "")
                    if name.lower().endswith((".htm", ".html")):
                        primary_doc = name
                        break

            if not primary_doc:
                raise ValueError(
                    f"No primary HTML document found for accession {accn}"
                )

        doc_url = f"{base}/{primary_doc}"
        time.sleep(_REQUEST_DELAY_SECONDS)
        with httpx.Client(headers=self._headers, timeout=self._timeout) as client:
            resp = client.get(doc_url)
            resp.raise_for_status()
            return resp.text
