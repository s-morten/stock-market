"""
EDGAR API client for fetching REIT company data from the SEC.

Uses the public SEC EDGAR REST API:
- Submissions endpoint: metadata about the company and its filings.
- Company Facts endpoint: XBRL financial data for all reported concepts.

The SEC requires a descriptive User-Agent header on every request.
All network calls use httpx for easy async upgrade and test mocking.

EDGAR API response logging
--------------------------
Every API response (URL, HTTP status, full JSON payload) is written to a
rotating log file so that all data received from the SEC can be inspected
offline.  The log path defaults to ``logs/edgar_api.log`` relative to the
working directory and can be overridden via the ``EDGAR_LOG_FILE``
environment variable.  Set ``EDGAR_LOG_ENABLED=0`` to disable file logging
without changing any code.
"""

import json
import logging
import logging.handlers
import os
import time
from datetime import date
from typing import Any

import httpx
from pydantic import BaseModel, Field

# Base URL for the EDGAR data API (not the search/EFTS endpoint).
_EDGAR_BASE = "https://data.sec.gov"

# GAAP concepts fetched for every company in the PoC (all denominated in USD).
POC_CONCEPTS = [
    # Core income statement / balance sheet
    "Revenues",
    "NetIncomeLoss",
    "Assets",
    "Liabilities",
    # Debt & interest
    "LongTermDebt",
    "LongTermDebtNoncurrent",
    "ShortTermBorrowings",
    "InterestExpense",
    "InterestAndDebtExpense",
    # Earnings per share (unit varies: "USD/shares" – use auto-detect)
    "EarningsPerShareBasic",
    "EarningsPerShareDiluted",
]

# Per-concept unit override for fetch_all_poc_facts.
# Concepts absent from this dict default to "USD".
# None triggers automatic unit selection (largest entry count).
POC_CONCEPT_UNITS: dict[str, str | None] = {
    "EarningsPerShareBasic": None,
    "EarningsPerShareDiluted": None,
}

# Minimum seconds between outgoing HTTP requests.
# The SEC enforces a limit of 10 requests/second; 0.11 s gives a safe margin.
_REQUEST_DELAY_SECONDS = 0.11

# ---------------------------------------------------------------------------
# File logger for EDGAR API responses
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)


def _build_edgar_file_logger() -> logging.Logger | None:
    """
    Create and return a dedicated file logger for raw EDGAR API responses.

    The logger writes one JSON record per line to a rotating log file so
    every response from the SEC API can be audited offline.

    Configuration via environment variables
    ----------------------------------------
    ``EDGAR_LOG_FILE``
        Path to the log file.  Defaults to ``logs/edgar_api.log`` relative
        to the current working directory.
    ``EDGAR_LOG_ENABLED``
        Set to ``0`` / ``false`` / ``no`` to disable file logging entirely
        (e.g. during unit tests).  Enabled by default.

    Returns:
        logging.Logger configured with a RotatingFileHandler, or ``None``
        if logging is disabled.
    """
    enabled = os.getenv("EDGAR_LOG_ENABLED", "1").strip().lower()
    if enabled in {"0", "false", "no", "off"}:
        return None

    log_path = os.getenv("EDGAR_LOG_FILE", "logs/edgar_api.log")
    log_file = os.path.abspath(log_path)

    # Create parent directory if needed.
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    file_logger = logging.getLogger("edgar_api_responses")
    file_logger.setLevel(logging.DEBUG)
    file_logger.propagate = False  # Don't pollute the root logger.

    # Only add the handler once (guard against repeated module imports).
    if not file_logger.handlers:
        handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=50 * 1024 * 1024,  # 50 MB per file
            backupCount=5,
            encoding="utf-8",
        )
        handler.setFormatter(logging.Formatter("%(message)s"))
        file_logger.addHandler(handler)

    return file_logger


# Module-level logger instance (created once on import).
_edgar_file_logger: logging.Logger | None = _build_edgar_file_logger()


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

    def _log_response(self, url: str, status_code: int, body: Any) -> None:
        """
        Write one JSON log record for an EDGAR API response.

        Each record contains:
        - ``timestamp`` – ISO-8601 UTC time of the response.
        - ``url``        – Full request URL (no credentials embedded).
        - ``status``     – HTTP status code.
        - ``body``       – Full parsed JSON response body.

        Parameters:
            url:         Request URL.
            status_code: HTTP response status code.
            body:        Parsed JSON body (dict or list).
        """
        if _edgar_file_logger is None:
            return

        from datetime import datetime, timezone

        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "url": url,
            "status": status_code,
            "body": body,
        }
        try:
            _edgar_file_logger.debug(json.dumps(record, default=str))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to write EDGAR log record: %s", exc)

    def _get(self, url: str) -> dict[str, Any]:
        """
        Perform a GET request and return the parsed JSON body.

        A fixed delay of ``_REQUEST_DELAY_SECONDS`` is observed *before*
        every request so the caller never exceeds the SEC rate limit of
        10 requests per second, even when many concepts are fetched in a
        tight loop.

        Every response (URL, status code, full JSON body) is written to
        the rotating EDGAR API log file via :meth:`_log_response`.

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
            # Log the raw response before raising so error responses are
            # captured too.
            try:
                body = response.json()
            except Exception:  # noqa: BLE001
                body = response.text
            self._log_response(url, response.status_code, body)
            logger.debug("EDGAR GET %s → %d", url, response.status_code)
            response.raise_for_status()
            return body  # type: ignore[return-value]

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
                        unit=POC_CONCEPT_UNITS.get(concept, "USD"),
                        since_date=since_date,
                        forms=forms,
                    )
                )
            except (httpx.HTTPStatusError, KeyError):
                # Concept not available for this company – skip gracefully.
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
                try:
                    index = resp.json()
                except Exception:  # noqa: BLE001
                    index = resp.text
                self._log_response(index_url, resp.status_code, index)
                logger.debug("EDGAR GET %s → %d", index_url, resp.status_code)
                resp.raise_for_status()

            # Find the primary document (.htm whose type matches the form).
            _FORM_TYPES = {"10-K", "10-Q", "10-K/A", "10-Q/A"}
            primary_doc = None
            for item in (index if isinstance(index, dict) else {}).get("documents", []):
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
            # Log metadata only for HTML responses (body is too large to store).
            self._log_response(
                doc_url,
                resp.status_code,
                {"content_type": resp.headers.get("content-type", ""),
                 "content_length_bytes": len(resp.content),
                 "note": "HTML body omitted from log (use raw filing URL above)"},
            )
            logger.debug("EDGAR GET %s → %d (%d bytes)", doc_url, resp.status_code, len(resp.content))
            resp.raise_for_status()
            return resp.text
