"""
Unit tests for reit_dashboard.data.property_parser.

Uses synthetic HTML fixtures to verify the parsing pipeline without any
network calls.
"""

import pytest

from reit_dashboard.data.property_parser import (
    PropertyRecord,
    PropertyTableResult,
    _clean_text,
    _normalise_metric,
    extract_property_data,
    find_item2_section,
    find_next_table,
    parse_property_table,
    property_result_to_rows,
)
from bs4 import BeautifulSoup


# ---------------------------------------------------------------------------
# Minimal HTML fixtures
# ---------------------------------------------------------------------------

_SIMPLE_HTML = """
<html><body>
  <h2>Item 2. Properties</h2>
  <table>
    <tr>
      <th>Property Type</th>
      <th>Number of Properties</th>
      <th>% Leased</th>
    </tr>
    <tr>
      <td>Industrial</td>
      <td>1,200</td>
      <td>97.5%</td>
    </tr>
    <tr>
      <td>Office</td>
      <td>45</td>
      <td>90.0%</td>
    </tr>
  </table>
</body></html>
"""

_NO_ITEM2_HTML = """
<html><body>
  <h2>Item 1. Business</h2>
  <table><tr><td>Data</td></tr></table>
</body></html>
"""

_ITEM2_NO_TABLE_HTML = """
<html><body>
  <h2>Item 2. Properties</h2>
  <p>We own properties in the US.</p>
</body></html>
"""

_UPPERCASE_ITEM2_HTML = """
<html><body>
  <h3>ITEM 2. PROPERTIES</h3>
  <table>
    <tr><th>Type</th><th>Sq. Ft.</th></tr>
    <tr><td>Multifamily</td><td>5,000,000</td></tr>
  </table>
</body></html>
"""

_FOOTNOTE_HTML = """
<html><body>
  <h2>Item 2 Properties</h2>
  <table>
    <tr><th>Type</th><th>Number of Properties</th></tr>
    <tr><td>Retail</td><td>500</td></tr>
    <tr><td>(1) Includes properties under development</td><td>20</td></tr>
    <tr><td>Total</td><td>520</td></tr>
  </table>
</body></html>
"""

_EMPTY_ROW_HTML = """
<html><body>
  <h2>Item 2. Properties</h2>
  <table>
    <tr><th>Type</th><th>Number of Properties</th></tr>
    <tr><td></td><td>100</td></tr>
    <tr><td>Warehouse</td><td>300</td></tr>
  </table>
</body></html>
"""


# ---------------------------------------------------------------------------
# Tests – text helpers
# ---------------------------------------------------------------------------

class TestCleanText:
    def test_collapses_whitespace(self):
        assert _clean_text("  foo   bar  ") == "foo bar"

    def test_replaces_nbsp(self):
        assert _clean_text("foo\xa0bar") == "foo bar"


class TestNormaliseMetric:
    def test_number_of_properties(self):
        assert _normalise_metric("Number of Properties") == "num_properties"

    def test_pct_leased(self):
        assert _normalise_metric("% Leased") == "pct_leased"

    def test_sqft(self):
        assert _normalise_metric("Sq. Ft.") == "sqft"

    def test_unknown_slugified(self):
        result = _normalise_metric("Custom Metric Column")
        assert result == "custom_metric_column"


# ---------------------------------------------------------------------------
# Tests – find_item2_section
# ---------------------------------------------------------------------------

class TestFindItem2Section:
    def test_finds_h2_item2(self):
        soup = BeautifulSoup(_SIMPLE_HTML, "lxml")
        tag = find_item2_section(soup)
        assert tag is not None
        assert "item 2" in tag.get_text().lower()

    def test_returns_none_when_absent(self):
        soup = BeautifulSoup(_NO_ITEM2_HTML, "lxml")
        assert find_item2_section(soup) is None

    def test_case_insensitive(self):
        soup = BeautifulSoup(_UPPERCASE_ITEM2_HTML, "lxml")
        assert find_item2_section(soup) is not None

    def test_no_period_in_heading(self):
        """'Item 2 Properties' without a period should still match."""
        soup = BeautifulSoup(_FOOTNOTE_HTML, "lxml")
        assert find_item2_section(soup) is not None


# ---------------------------------------------------------------------------
# Tests – parse_property_table
# ---------------------------------------------------------------------------

class TestParsePropertyTable:
    def _table(self, html: str) -> BeautifulSoup:
        return BeautifulSoup(html, "lxml").find("table")

    def test_parses_two_data_rows(self):
        table = self._table(_SIMPLE_HTML)
        result = parse_property_table(table, "0001045609-23-000001", "10-K")
        assert len(result.records) == 2

    def test_header_extraction(self):
        table = self._table(_SIMPLE_HTML)
        result = parse_property_table(table, "acc", "10-K")
        assert "num_properties" in result.headers or "Number of Properties" in result.headers

    def test_property_type_labels(self):
        table = self._table(_SIMPLE_HTML)
        result = parse_property_table(table, "acc", "10-K")
        types = [r.property_type for r in result.records]
        assert "Industrial" in types
        assert "Office" in types

    def test_numeric_value_stripped_of_commas(self):
        table = self._table(_SIMPLE_HTML)
        result = parse_property_table(table, "acc", "10-K")
        industrial = next(r for r in result.records if r.property_type == "Industrial")
        assert industrial.metrics["num_properties"] == "1200"

    def test_footnote_rows_skipped(self):
        """Rows starting with '(' or a digit should be excluded."""
        table = self._table(_FOOTNOTE_HTML)
        result = parse_property_table(table, "acc", "10-K")
        types = [r.property_type for r in result.records]
        assert all(not t.startswith("(") for t in types)

    def test_empty_label_row_skipped(self):
        table = self._table(_EMPTY_ROW_HTML)
        result = parse_property_table(table, "acc", "10-K")
        # Only "Warehouse" row should appear.
        assert len(result.records) == 1
        assert result.records[0].property_type == "Warehouse"

    def test_sqft_alias(self):
        table = self._table(_UPPERCASE_ITEM2_HTML)
        result = parse_property_table(table, "acc", "10-K")
        assert result.records[0].metrics.get("sqft") is not None


# ---------------------------------------------------------------------------
# Tests – extract_property_data (full pipeline)
# ---------------------------------------------------------------------------

class TestExtractPropertyData:
    def test_full_pipeline_happy_path(self):
        result = extract_property_data(_SIMPLE_HTML, "acc-001", "10-K")
        assert isinstance(result, PropertyTableResult)
        assert len(result.records) == 2
        assert result.accn == "acc-001"
        assert result.form == "10-K"

    def test_no_item2_returns_empty(self):
        result = extract_property_data(_NO_ITEM2_HTML, "acc-002", "10-K")
        assert result.records == []

    def test_item2_no_table_returns_empty(self):
        result = extract_property_data(_ITEM2_NO_TABLE_HTML, "acc-003", "10-K")
        assert result.records == []

    def test_uppercase_heading_handled(self):
        result = extract_property_data(_UPPERCASE_ITEM2_HTML, "acc-004", "10-K")
        assert len(result.records) == 1


# ---------------------------------------------------------------------------
# Tests – property_result_to_rows
# ---------------------------------------------------------------------------

class TestPropertyResultToRows:
    def test_flattens_to_rows(self):
        result = extract_property_data(_SIMPLE_HTML, "acc-001", "10-K")
        rows = property_result_to_rows(result, "0001045609", "PLD", "2022-12-31")
        # 2 property types × 2 metrics = 4 rows
        assert len(rows) == 4

    def test_row_schema(self):
        result = extract_property_data(_SIMPLE_HTML, "acc-001", "10-K")
        rows = property_result_to_rows(result, "0001045609", "PLD", "2022-12-31")
        for row in rows:
            assert "cik" in row
            assert "ticker" in row
            assert "accn" in row
            assert "period_end" in row
            assert "property_type" in row
            assert "metric_name" in row
            assert "value" in row

    def test_empty_result_gives_no_rows(self):
        result = extract_property_data(_NO_ITEM2_HTML, "acc-002", "10-K")
        rows = property_result_to_rows(result, "0001045609", "PLD", "2022-12-31")
        assert rows == []


# ---------------------------------------------------------------------------
# Tests for iXBRL detection and parser selection
# ---------------------------------------------------------------------------

_IXBRL_HTML = """\
<?xml version="1.0" encoding="UTF-8"?>
<html xmlns="http://www.w3.org/1999/xhtml"
      xmlns:ix="https://xbrl.org/2013/inlineXBRL">
<body>
<p>ITEM 2. Properties</p>
<div>
  <table>
    <tr><th></th><th>Number of Properties</th></tr>
    <tr><td>Industrial</td><td>500</td></tr>
    <tr><td>Office</td><td>120</td></tr>
  </table>
</div>
</body>
</html>
"""

_PLAIN_HTML = """\
<html>
<body>
<h2>Item 2. Properties</h2>
<table>
  <tr><th></th><th>Number of Properties</th></tr>
  <tr><td>Industrial</td><td>300</td></tr>
</table>
</body>
</html>
"""


class TestIXBRLDetection:
    """Tests for _detect_parser and iXBRL-aware extract_property_data."""

    def test_detects_xml_declaration(self):
        """<?xml version …> prefix triggers lxml-xml parser."""
        from reit_dashboard.data.property_parser import _detect_parser

        assert _detect_parser('<?xml version="1.0"?><html>') == "lxml-xml"

    def test_detects_ix_namespace(self):
        """xmlns:ix= attribute triggers lxml-xml parser."""
        from reit_dashboard.data.property_parser import _detect_parser

        assert _detect_parser('<html xmlns:ix="https://xbrl.org/2013">') == "lxml-xml"

    def test_plain_html_uses_lxml(self):
        """Plain HTML without XML markers uses the lxml HTML parser."""
        from reit_dashboard.data.property_parser import _detect_parser

        assert _detect_parser("<html><body><h1>Hello</h1></body></html>") == "lxml"

    def test_ixbrl_item2_found_via_p_tag(self):
        """iXBRL document where Item 2 is in a <p> tag is parsed correctly."""
        result = extract_property_data(_IXBRL_HTML, "acc-ixbrl-001", "10-K")
        assert len(result.records) == 2
        types = [r.property_type for r in result.records]
        assert "Industrial" in types
        assert "Office" in types

    def test_ixbrl_num_properties_extracted(self):
        """num_properties metric is extracted from an iXBRL table."""
        result = extract_property_data(_IXBRL_HTML, "acc-ixbrl-001", "10-K")
        industrial = next(r for r in result.records if r.property_type == "Industrial")
        assert industrial.metrics.get("num_properties") == "500"

    def test_plain_html_still_works(self):
        """Ensure the non-iXBRL code path is not broken."""
        result = extract_property_data(_PLAIN_HTML, "acc-plain-001", "10-K")
        assert len(result.records) == 1
        assert result.records[0].property_type == "Industrial"
