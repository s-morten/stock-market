"""
EDGAR API client for fetching REIT company data from the SEC.

Uses the public SEC EDGAR REST API:
- Submissions endpoint: metadata about the company and its filings.
- Company Facts endpoint: XBRL financial data for all reported concepts.

The SEC requires a descriptive User-Agent header on every request.
All network calls use httpx for easy async upgrade and test mocking.
"""

import time
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import httpx
from pydantic import BaseModel, Field

# Base URL for the EDGAR data API (not the search/EFTS endpoint).
_EDGAR_BASE = "https://data.sec.gov"

# Minimum seconds between outgoing HTTP requests.
# The SEC enforces a limit of 10 requests/second; 0.11 s gives a safe margin.
_REQUEST_DELAY_SECONDS = 0.11


@dataclass(frozen=True)
class ConceptConfig:
    """
    Configuration for a single XBRL concept to fetch.

    Attributes:
        concept:   XBRL tag name (e.g. "Revenues").
        unit:      Expected unit string (e.g. "USD", "USD/shares", "sqft").
                   When None the client auto-detects the first available unit.
        forms:     Set of SEC filing form types to accept.  Defaults to
                   both annual and quarterly filings.
        taxonomy:  XBRL taxonomy namespace.  Almost always "us-gaap".
    """

    concept: str
    unit: str | None = None
    forms: frozenset[str] = field(
        default_factory=lambda: frozenset({"10-K", "10-Q"})
    )
    taxonomy: str = "us-gaap"


# ---------------------------------------------------------------------------
# Concept catalogue
# ---------------------------------------------------------------------------

# Core quarterly financial concepts (existing).
CORE_CONCEPTS: list[ConceptConfig] = [
    ConceptConfig("Revenues", "USD"),
    ConceptConfig("NetIncomeLoss", "USD"),
    ConceptConfig("Assets", "USD"),
    ConceptConfig("Liabilities", "USD"),
]

# Extended concepts added for richer REIT analysis.
EXTENDED_CONCEPTS: list[ConceptConfig] = [
    # --- FFO components ---
    ConceptConfig("DepreciationAndAmortization", "USD"),
    # Two alternative tags for property sale gains; both are tried.
    ConceptConfig("GainLossOnSaleOfProperties", "USD"),
    ConceptConfig("GainsLossesOnSalesOfInvestmentRealEstate", "USD"),
    # --- Additional income statement / cash flow ---
    ConceptConfig("OperatingIncomeLoss", "USD"),
    ConceptConfig("InterestExpense", "USD"),
    ConceptConfig("NetCashProvidedByUsedInOperatingActivities", "USD"),
    # --- Portfolio / balance sheet ---
    ConceptConfig("RealEstateInvestmentPropertyNet", "USD"),
    # NumberOfRealEstateProperties uses a non-standard unit; auto-detect.
    ConceptConfig("NumberOfRealEstateProperties", None),
    # AreaOfRealEstateProperty is reported in sqft or sqmt; auto-detect.
    ConceptConfig("AreaOfRealEstateProperty", None),
    # --- Dividends ---
    # Reported per-share; unit is "USD/shares".
    ConceptConfig("CommonStockDividendsPerShareDeclared", "USD/shares"),
    # --- Debt ---
    ConceptConfig("LongTermDebt", "USD"),
    # Debt maturity schedule by year – annual filings only.
    ConceptConfig(
        "LongTermDebtMaturitiesRepaymentsOfPrincipalInNextTwelveMonths",
        "USD",
        frozenset({"10-K"}),
    ),
    ConceptConfig(
        "LongTermDebtMaturitiesRepaymentsOfPrincipalInYearTwo",
        "USD",
        frozenset({"10-K"}),
    ),
    ConceptConfig(
        "LongTermDebtMaturitiesRepaymentsOfPrincipalInYearThree",
        "USD",
        frozenset({"10-K"}),
    ),
    ConceptConfig(
        "LongTermDebtMaturitiesRepaymentsOfPrincipalInYearFour",
        "USD",
        frozenset({"10-K"}),
    ),
    ConceptConfig(
        "LongTermDebtMaturitiesRepaymentsOfPrincipalInYearFive",
        "USD",
        frozenset({"10-K"}),
    ),
    ConceptConfig(
        "LongTermDebtMaturitiesRepaymentsOfPrincipalAfterYearFive",
        "USD",
        frozenset({"10-K"}),
    ),
]

# Convenience: all concepts in one list.
ALL_CONCEPTS: list[ConceptConfig] = CORE_CONCEPTS + EXTENDED_CONCEPTS

# Legacy list kept for backwards-compatibility with existing tests.
POC_CONCEPTS: list[str] = [c.concept for c in CORE_CONCEPTS]


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

        When ``unit`` is ``None`` the method auto-detects the first unit
        with data (useful for concepts like NumberOfRealEstateProperties
        that use non-standard unit strings).

        Parameters:
            cik:        Company CIK.
            concept:    GAAP XBRL tag (e.g. "Revenues").
            taxonomy:   XBRL taxonomy namespace (default "us-gaap").
            unit:       Unit of measure to extract (default "USD").
                        Pass ``None`` to auto-detect the first available
                        unit.
            since_date: Exclude entries whose period end is before this
                        date.  Pass ``None`` to include all history.
            forms:      Set of filing form types to keep.  Defaults to
                        ``{"10-K", "10-Q"}``.

        Returns:
            ConceptFacts: Parsed fact entries for the concept.

        Raises:
            KeyError: If the concept or no units are present in the data.
        """
        if forms is None:
            forms = {"10-K", "10-Q"}

        padded = self._pad_cik(cik)
        url = (
            f"{_EDGAR_BASE}/api/xbrl/companyconcept/"
            f"CIK{padded}/{taxonomy}/{concept}.json"
        )
        data = self._get(url)

        units_data: dict[str, Any] = data.get("units", {})
        if not units_data:
            raise KeyError(f"No units found for concept {concept!r}")

        # Auto-detect unit if not specified.
        resolved_unit: str
        if unit is None:
            resolved_unit = next(iter(units_data))
        else:
            resolved_unit = unit

        raw_entries: list[dict[str, Any]] = units_data.get(resolved_unit, [])

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
        return ConceptFacts(concept=concept, unit=resolved_unit, entries=entries)

    def fetch_all_poc_facts(
        self,
        cik: str,
        since_date: date | None = None,
        forms: set[str] | None = None,
    ) -> list[ConceptFacts]:
        """
        Fetch the four core XBRL concepts for a company (legacy helper).

        Silently skips any concept that is not reported by the company.

        Parameters:
            cik:        Company CIK.
            since_date: Earliest period end date to include.
            forms:      Filing form types to include.

        Returns:
            list[ConceptFacts]: One entry per successfully fetched concept.
        """
        return self.fetch_concepts(
            cik,
            CORE_CONCEPTS,
            since_date=since_date,
            forms_override=forms,
        )

    def fetch_concepts(
        self,
        cik: str,
        configs: list[ConceptConfig],
        since_date: date | None = None,
        forms_override: set[str] | None = None,
    ) -> list[ConceptFacts]:
        """
        Fetch an arbitrary list of XBRL concepts for a company.

        Each concept in ``configs`` carries its own unit hint and forms
        filter.  ``forms_override`` replaces the per-concept forms setting
        when provided (useful for bulk overrides such as "10-Q only").

        Silently skips concepts that return a 404 or have no matching data.

        Parameters:
            cik:            Company CIK.
            configs:        List of ConceptConfig descriptors.
            since_date:     Earliest period end date to include.
            forms_override: When set, overrides the forms in every config.

        Returns:
            list[ConceptFacts]: One entry per successfully fetched concept.
        """
        results: list[ConceptFacts] = []
        for cfg in configs:
            effective_forms = (
                set(forms_override) if forms_override is not None
                else set(cfg.forms)
            )
            try:
                cf = self.fetch_concept_facts(
                    cik,
                    cfg.concept,
                    taxonomy=cfg.taxonomy,
                    unit=cfg.unit,
                    since_date=since_date,
                    forms=effective_forms,
                )
                if cf.entries:
                    results.append(cf)
            except (httpx.HTTPStatusError, KeyError):
                # Concept not available for this company – skip gracefully.
                pass
        return results
