"""
Parser for REIT property data extracted from SEC 10-K/10-Q filings.

Workflow
--------
1. Find the "Item 2. Properties" section in the filing HTML by searching
   for a heading tag (h1–h3) whose text matches the expected pattern.
2. Locate the first ``<table>`` that follows that heading.
3. Parse the table rows into a structured dict:
   ``{property_type: {metric_name: value, ...}, ...}``

The parser is intentionally tolerant of formatting variations between
companies and across filing years.  Modern filings use iXBRL (Inline XBRL),
which is an XML document.  The parser detects iXBRL automatically and
selects the appropriate BeautifulSoup parser (``lxml-xml`` for iXBRL,
``lxml`` for plain HTML).
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from typing import Any

from bs4 import BeautifulSoup, NavigableString, Tag, XMLParsedAsHTMLWarning

# Regex that matches the "Item 2" heading in various formats:
#   "Item 2. Properties"
#   "Item 2. Properties."
#   "ITEM 2. PROPERTIES"
#   "Item 2 Properties"  (no period)
_ITEM2_PATTERN = re.compile(
    r"item\s+2[\.\s]+propert",
    re.IGNORECASE,
)

# Headings we scan for the Item 2 marker.
_HEADING_TAGS = ("h1", "h2", "h3", "h4")

# Fallback inline/block tags used when no heading-level tag is found
# (common in iXBRL filings and older plain-HTML 10-K documents).
_FALLBACK_TAGS = ("b", "strong", "p", "span", "div")

# iXBRL indicator: modern Inline XBRL documents declare the ix: namespace.
_IXBRL_PATTERN = re.compile(
    r'xmlns:ix\s*=|<\?xml\s+version',
    re.IGNORECASE,
)


def _detect_parser(html: str) -> str:
    """
    Choose the appropriate BeautifulSoup parser for the filing HTML.

    Modern SEC filings are Inline XBRL (iXBRL) documents.  These are
    XML at the wire level and must be parsed with ``lxml-xml`` to
    correctly resolve the element tree.  Plain HTML filings (older
    documents) use the ``lxml`` HTML parser.

    Parameters:
        html: Raw text content of the filing document.

    Returns:
        ``"lxml-xml"`` for iXBRL/XML documents, ``"lxml"`` otherwise.
    """
    # Only scan the first 2 kB – the declaration is always near the top.
    return "lxml-xml" if _IXBRL_PATTERN.search(html[:2048]) else "lxml"

# Common metric column names and their normalised keys.
_METRIC_ALIASES: dict[str, str] = {
    "number of properties": "num_properties",
    "# of properties": "num_properties",
    "no. of properties": "num_properties",
    "number of buildings": "num_properties",
    "number of assets": "num_properties",
    "total properties": "num_properties",
    "% leased": "pct_leased",
    "percent leased": "pct_leased",
    "% occupied": "pct_leased",
    "occupancy": "pct_leased",
    "occupancy rate": "pct_leased",
    "occupancy (%)": "pct_leased",
    "square feet": "sqft",
    "sq. ft.": "sqft",
    "sq ft": "sqft",
    "rentable square feet": "sqft",
    "gross leasable area": "sqft",
    "gla (sq. ft.)": "sqft",
    "annualized base rent": "annualized_base_rent",
    "annualized rent": "annualized_base_rent",
    "abr": "annualized_base_rent",
    "net book value": "net_book_value",
    "book value": "net_book_value",
    "carrying value": "net_book_value",
}


@dataclass
class PropertyRecord:
    """
    A single row of parsed property data from a filing.

    Attributes:
        property_type: The row label (e.g. "Office", "Multifamily").
        metrics:       Mapping of normalised metric key → raw string value.
    """

    property_type: str
    metrics: dict[str, str] = field(default_factory=dict)


@dataclass
class PropertyTableResult:
    """
    Result of parsing the Item 2 property table from one filing.

    Attributes:
        accn:       Accession number of the source filing.
        form:       Filing form type (e.g. "10-K").
        records:    Parsed property rows.  Empty when parsing failed.
        headers:    Column headers extracted from the table.
        raw_html:   The outer HTML of the matched table (for debugging).
    """

    accn: str
    form: str
    records: list[PropertyRecord] = field(default_factory=list)
    headers: list[str] = field(default_factory=list)
    raw_html: str = ""


def _clean_text(text: str) -> str:
    """Collapse whitespace and strip unicode non-breaking spaces."""
    return re.sub(r"\s+", " ", text.replace("\xa0", " ")).strip()


def _normalise_metric(header: str) -> str:
    """Map a raw column header to a canonical metric key."""
    lower = header.lower().strip()
    for alias, key in _METRIC_ALIASES.items():
        if alias in lower:
            return key
    # Fall back to slugified header.
    return re.sub(r"[^a-z0-9]+", "_", lower).strip("_")


def _parse_numeric(text: str) -> str:
    """
    Return the text with light normalisation (removes $ and commas).

    Keeps percent signs and the original value as a string so callers
    decide how to cast the data.
    """
    return text.replace("$", "").replace(",", "").strip()


def find_item2_section(soup: BeautifulSoup) -> Tag | None:
    """
    Locate the heading element that marks the start of Item 2.

    Searches all h1–h4 tags for text matching :data:`_ITEM2_PATTERN`.
    Falls back to :data:`_FALLBACK_TAGS` (bold, paragraph, span, div)
    which cover older plain-HTML filings and modern iXBRL documents
    where Item 2 is rendered in a styled ``<p>`` or ``<span>``.

    The match is restricted to short text snippets (≤ 200 chars) so we
    don't accidentally match long paragraphs that merely mention
    "Item 2".

    Parameters:
        soup: Parsed BeautifulSoup document.

    Returns:
        The matching Tag, or ``None`` if not found.
    """
    for tag in soup.find_all(_HEADING_TAGS):
        text = _clean_text(tag.get_text())
        if _ITEM2_PATTERN.search(text) and len(text) <= 200:
            return tag
    # Fallback: bold/strong/paragraph/span/div acting as headings in
    # older plain-HTML filings and iXBRL-based modern filings.
    for tag in soup.find_all(_FALLBACK_TAGS):
        text = _clean_text(tag.get_text())
        if _ITEM2_PATTERN.search(text) and len(text) <= 200:
            return tag
    return None


def find_next_table(start_tag: Tag) -> Tag | None:
    """
    Walk the DOM forward from *start_tag* to find the next ``<table>``.

    Searches siblings first, then climbs to the parent and searches its
    subsequent siblings, up to 3 levels.

    Parameters:
        start_tag: The element from which to begin the forward search.

    Returns:
        The first ``<table>`` tag found, or ``None``.
    """
    node: Tag | NavigableString | None = start_tag
    for _ in range(3):  # climb at most 3 parent levels
        sibling = node.find_next_sibling() if hasattr(node, "find_next_sibling") else None
        while sibling is not None:
            if isinstance(sibling, Tag):
                if sibling.name == "table":
                    return sibling
                # Table may be nested one level deep in a div/section.
                found = sibling.find("table")
                if found:
                    return found
            sibling = sibling.find_next_sibling() if hasattr(sibling, "find_next_sibling") else None
        # No table found among siblings; climb up.
        node = getattr(node, "parent", None)
        if node is None:
            break
    return None


def parse_property_table(
    table: Tag,
    accn: str,
    form: str,
) -> PropertyTableResult:
    """
    Parse an HTML ``<table>`` element into :class:`PropertyTableResult`.

    Extracts the header row as column names and each subsequent row as a
    :class:`PropertyRecord`.  Rows whose first cell is empty or numeric
    (e.g. footnote rows) are skipped.

    Parameters:
        table: The ``<table>`` BeautifulSoup Tag.
        accn:  Accession number (stored in the result for traceability).
        form:  Filing form type.

    Returns:
        :class:`PropertyTableResult` populated with parsed records.
    """
    result = PropertyTableResult(
        accn=accn,
        form=form,
        raw_html=str(table)[:2000],  # truncate for storage
    )

    rows = table.find_all("tr")
    if not rows:
        return result

    # --- Extract header row ---
    header_row = rows[0]
    raw_headers = [
        _clean_text(th.get_text())
        for th in header_row.find_all(["th", "td"])
    ]
    # Drop leading empty header (often the "Property Type" label column).
    result.headers = raw_headers

    if len(raw_headers) < 2:
        return result

    # Column 0 is the row label; columns 1+ are metrics.
    metric_headers = [_normalise_metric(h) for h in raw_headers[1:]]

    # --- Parse data rows ---
    for row in rows[1:]:
        cells = row.find_all(["td", "th"])
        if not cells:
            continue

        label = _clean_text(cells[0].get_text())
        if not label:
            continue
        # Skip footnote / total rows that start with a digit or * or (1).
        if re.match(r"^[\d\*\(\[]", label):
            continue

        record = PropertyRecord(property_type=label)
        for i, metric_key in enumerate(metric_headers, start=1):
            if i >= len(cells):
                break
            raw_val = _clean_text(cells[i].get_text())
            if raw_val and raw_val not in ("-", "—", "N/A", "n/a"):
                record.metrics[metric_key] = _parse_numeric(raw_val)

        if record.metrics:
            result.records.append(record)

    return result


def extract_property_data(
    html: str,
    accn: str,
    form: str,
) -> PropertyTableResult:
    """
    Full pipeline: parse HTML → find Item 2 heading → parse first table.

    Automatically detects whether the document is an Inline XBRL (iXBRL)
    file and selects the appropriate parser (``lxml-xml`` for iXBRL,
    ``lxml`` for plain HTML).

    Parameters:
        html:  Raw HTML text of the SEC filing document.
        accn:  Accession number for attribution.
        form:  Filing form type.

    Returns:
        :class:`PropertyTableResult`.  ``records`` is empty when parsing
        failed (e.g. Item 2 or a following table was not found).
    """
    parser = _detect_parser(html)
    # Suppress BeautifulSoup's warning when we intentionally parse XML with
    # the HTML-mode lxml parser (only happens for the "lxml" fallback path).
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
        soup = BeautifulSoup(html, parser)

    item2_tag = find_item2_section(soup)
    if item2_tag is None:
        return PropertyTableResult(accn=accn, form=form)

    table = find_next_table(item2_tag)
    if table is None:
        return PropertyTableResult(accn=accn, form=form)

    return parse_property_table(table, accn, form)


def property_result_to_rows(
    result: PropertyTableResult,
    cik: str,
    ticker: str | None,
    period_end: str,
) -> list[dict[str, Any]]:
    """
    Flatten a :class:`PropertyTableResult` into a list of DB-ready dicts.

    Each dict represents one (property_type, metric_name, value) triple.

    Parameters:
        result:     Parsed property table result.
        cik:        Company CIK.
        ticker:     Ticker symbol (may be ``None``).
        period_end: ISO date string for the filing period end.

    Returns:
        list of dicts with keys: cik, ticker, accn, period_end, form,
        property_type, metric_name, value.
    """
    rows: list[dict[str, Any]] = []
    for record in result.records:
        for metric_name, raw_value in record.metrics.items():
            rows.append(
                {
                    "cik": cik,
                    "ticker": ticker,
                    "accn": result.accn,
                    "period_end": period_end,
                    "form": result.form,
                    "property_type": record.property_type,
                    "metric_name": metric_name,
                    "value": raw_value,
                }
            )
    return rows
