"""
Standalone script – run the Item 2 HTML property parser for all PoC REITs,
optionally call Gemini to extract total property counts, and write results
to output files.

No database interaction; results go to local files so you can inspect
what the parser and Gemini find without touching the database.

Usage
-----
    uv run python scripts/extract_property_data.py

Output
------
    property_extract.json  – structured results, one entry per filing
    property_extract.txt   – human-readable summary

Options (environment variables)
--------------------------------
    EDGAR_USER_AGENT   Required – your "Name email@example.com" identifier.
    GEMINI_API_KEY     Optional – enables Gemini property count extraction.
    EXTRACT_OUTPUT     Output file path (default: property_extract.json).
    EXTRACT_FORMS      Comma-separated form types (default: 10-K).
    EXTRACT_MAX        Max filings per company (default: 5).
"""

import json
import sys
from datetime import date, timedelta
from pathlib import Path

# Allow running from the repo root without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import os

from reit_dashboard.config import get_edgar_user_agent, get_gemini_api_key
from reit_dashboard.data.edgar_client import EdgarClient
from reit_dashboard.data.ingestion import POC_REITS, POC_TICKERS
from reit_dashboard.data.property_parser import (
    extract_item2_tables_text,
    extract_property_data,
    property_result_to_rows,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OUTPUT_PATH = Path(os.getenv("EXTRACT_OUTPUT", "property_extract.json"))
FORMS = set(os.getenv("EXTRACT_FORMS", "10-K").split(","))
MAX_FILINGS = int(os.getenv("EXTRACT_MAX", "5"))
SINCE_DATE = date.today() - timedelta(days=5 * 365)


def _print(msg: str) -> None:
    """Print to stdout and flush immediately."""
    print(msg, flush=True)


def run_extraction() -> list[dict]:
    """
    Fetch filings for all PoC REITs, run the HTML parser and optionally
    Gemini extraction, and return a list of structured result dicts.

    Returns:
        list[dict]: One entry per company, with ``filings`` list containing
            per-filing results including optional ``gemini_count``.
    """
    user_agent = get_edgar_user_agent()
    client = EdgarClient(user_agent=user_agent)

    # Set up Gemini extractor if API key is available.
    gemini_extractor = None
    gemini_key = get_gemini_api_key()
    if gemini_key:
        from reit_dashboard.data.gemini_client import GeminiPropertyExtractor
        gemini_extractor = GeminiPropertyExtractor(api_key=gemini_key)
        _print("Gemini API key found – property count extraction enabled.")
    else:
        _print("No GEMINI_API_KEY set – skipping Gemini extraction.")

    all_results: list[dict] = []

    for cik, name in POC_REITS.items():
        ticker = POC_TICKERS.get(cik, "")
        _print(f"\n{'=' * 60}")
        _print(f"  {name} ({ticker})  CIK: {cik}")
        _print(f"{'=' * 60}")

        try:
            filings = client.list_filings(
                cik,
                forms=FORMS,
                since_date=SINCE_DATE,
                max_filings=MAX_FILINGS,
            )
        except Exception as exc:  # noqa: BLE001
            _print(f"  ERROR listing filings: {exc}")
            all_results.append({
                "ticker": ticker,
                "cik": cik,
                "name": name,
                "error": str(exc),
                "filings": [],
            })
            continue

        _print(f"  Found {len(filings)} filing(s) to process")
        company_results: list[dict] = []

        for filing in filings:
            accn = filing["accn"]
            form = filing["form"]
            period_end = filing.get("reportDate") or filing.get("filingDate", "")
            primary_doc = filing.get("primaryDocument") or None

            _print(f"\n  [{form}] {period_end}  accn={accn}")
            if primary_doc:
                _print(f"    doc: {primary_doc}")

            entry: dict = {
                "ticker": ticker,
                "cik": cik,
                "accn": accn,
                "form": form,
                "period_end": period_end,
                "primary_doc": primary_doc,
                "status": "ok",
                "records_found": 0,
                "gemini_count": None,
                "gemini_notes": None,
                "rows": [],
            }

            # Fetch HTML
            try:
                html = client.fetch_filing_html(cik, accn, primary_doc=primary_doc)
                _print(f"    fetched HTML: {len(html):,} chars")
            except Exception as exc:  # noqa: BLE001
                _print(f"    FETCH ERROR: {exc}")
                entry["status"] = f"fetch_error: {exc}"
                company_results.append(entry)
                continue

            # Gemini property count extraction
            if gemini_extractor is not None:
                tables_text = extract_item2_tables_text(html)
                if tables_text:
                    try:
                        _print("    GEMINI: sending tables to Gemini...")
                        gemini_result = gemini_extractor.extract_property_count(tables_text)
                        entry["gemini_count"] = gemini_result.get("total_properties")
                        entry["gemini_notes"] = gemini_result.get("notes", "")
                        _print(
                            f"    GEMINI: total_properties={entry['gemini_count']} "
                            f"– {entry['gemini_notes']}"
                        )
                    except Exception as exc:  # noqa: BLE001
                        _print(f"    GEMINI ERROR: {exc}")
                        entry["gemini_notes"] = f"error: {exc}"
                else:
                    _print("    GEMINI: no Item 2 tables found to send")

            # Parse Item 2 rows (structural breakdown)
            result = extract_property_data(html, accn, form)
            entry["records_found"] = len(result.records)
            entry["headers"] = result.headers

            if not result.records:
                _print("    PARSE: no property table found")
                entry["status"] = "no_table"
            else:
                _print(f"    PARSE: {len(result.records)} property row(s) found")
                rows = property_result_to_rows(result, cik, ticker, period_end)
                entry["rows"] = rows

            company_results.append(entry)

        all_results.append({
            "ticker": ticker,
            "cik": cik,
            "name": name,
            "error": None,
            "filings": company_results,
        })

    return all_results


def _write_summary(results: list[dict], txt_path: Path) -> None:
    """Write a plain-text summary table to *txt_path*."""
    lines = ["Property Extraction Summary", "=" * 70, ""]
    total_filings = 0
    total_parsed = 0
    total_rows = 0
    total_gemini = 0

    for company in results:
        name = company["name"]
        ticker = company["ticker"]
        filings = company.get("filings", [])
        parsed = [f for f in filings if f.get("records_found", 0) > 0]
        rows = sum(len(f.get("rows", [])) for f in filings)
        gemini_hits = [f for f in filings if f.get("gemini_count") is not None]

        lines.append(f"{name} ({ticker})")
        lines.append(f"  Filings processed : {len(filings)}")
        lines.append(f"  Filings parsed    : {len(parsed)}")
        lines.append(f"  Gemini extractions: {len(gemini_hits)}")
        lines.append(f"  Data rows         : {rows}")

        for f in filings:
            if f.get("gemini_count") is not None or f.get("records_found", 0) > 0:
                gemini_str = (
                    f"  Gemini={f['gemini_count']}" if f.get("gemini_count") is not None else ""
                )
                lines.append(
                    f"  [{f['form']}] {f['period_end']}"
                    f"  ({f.get('records_found', 0)} rows){gemini_str}"
                )
                if f.get("gemini_notes"):
                    lines.append(f"    note: {f['gemini_notes']}")
        lines.append("")

        total_filings += len(filings)
        total_parsed += len(parsed)
        total_rows += rows
        total_gemini += len(gemini_hits)

    lines += [
        "-" * 70,
        f"Total filings processed : {total_filings}",
        f"Total filings parsed    : {total_parsed}",
        f"Total Gemini extractions: {total_gemini}",
        f"Total data rows         : {total_rows}",
    ]
    txt_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """Entry point: run extraction, write JSON + text summary."""
    _print(
        f"Starting property data extraction\n"
        f"  Forms  : {', '.join(sorted(FORMS))}\n"
        f"  Max/co : {MAX_FILINGS}\n"
        f"  Since  : {SINCE_DATE}\n"
        f"  Output : {OUTPUT_PATH}"
    )

    results = run_extraction()

    # Write JSON
    OUTPUT_PATH.write_text(
        json.dumps(results, indent=2, default=str),
        encoding="utf-8",
    )
    _print(f"\n\nJSON written → {OUTPUT_PATH}")

    # Write plain-text summary alongside the JSON
    txt_path = OUTPUT_PATH.with_suffix(".txt")
    _write_summary(results, txt_path)
    _print(f"Summary  written → {txt_path}")

    # Quick totals to stdout
    total_rows = sum(
        len(f.get("rows", []))
        for co in results
        for f in co.get("filings", [])
    )
    _print(f"\nTotal data rows extracted: {total_rows}")


if __name__ == "__main__":
    main()
