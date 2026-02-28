"""
Unit tests for reit_dashboard.data.stock_price_client.

yfinance.Ticker is patched with unittest.mock so no real HTTP calls are made.
time.sleep is patched to keep tests fast.
"""

from datetime import date
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from reit_dashboard.data.stock_price_client import StockPriceClient, StockPriceEntry

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_history_df() -> pd.DataFrame:
    """Build a minimal fake yfinance history DataFrame."""
    index = pd.to_datetime(["2024-01-01", "2024-01-08", "2024-01-15"])
    return pd.DataFrame(
        {
            "Open":   [100.0, 102.0, 98.0],
            "High":   [105.0, 107.0, 103.0],
            "Low":    [98.0,  100.0, 96.0],
            "Close":  [103.0, 105.0, 99.0],
            "Volume": [1_000_000, 1_100_000, 900_000],
        },
        index=index,
    )


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    """Patch time.sleep so rate-limit delays don't slow down tests."""
    monkeypatch.setattr(
        "reit_dashboard.data.stock_price_client.time.sleep",
        lambda _: None,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestFetchWeeklyPrices:
    """Tests for StockPriceClient.fetch_weekly_prices."""

    def test_returns_sorted_entries(self):
        """Returned entries are sorted by date ascending."""
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = _make_history_df()

        with patch(
            "reit_dashboard.data.stock_price_client.yf.Ticker",
            return_value=mock_ticker,
        ):
            client = StockPriceClient()
            entries = client.fetch_weekly_prices("PLD")

        assert len(entries) == 3
        assert entries[0].date < entries[1].date < entries[2].date

    def test_entry_fields_are_populated(self):
        """Every StockPriceEntry has the expected numeric fields."""
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = _make_history_df()

        with patch(
            "reit_dashboard.data.stock_price_client.yf.Ticker",
            return_value=mock_ticker,
        ):
            client = StockPriceClient()
            entries = client.fetch_weekly_prices("PLD")

        entry = entries[0]
        assert isinstance(entry, StockPriceEntry)
        assert entry.open == 100.0
        assert entry.high == 105.0
        assert entry.low == 98.0
        assert entry.close == 103.0
        assert entry.volume == 1_000_000

    def test_since_date_passed_to_history(self):
        """since_date is forwarded to yfinance as the 'start' kwarg."""
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = _make_history_df()

        with patch(
            "reit_dashboard.data.stock_price_client.yf.Ticker",
            return_value=mock_ticker,
        ):
            client = StockPriceClient()
            client.fetch_weekly_prices("PLD", since_date=date(2023, 1, 1))

        _, kwargs = mock_ticker.history.call_args
        assert kwargs["start"] == "2023-01-01"
        assert kwargs["interval"] == "1wk"

    def test_no_since_date_uses_5y_period(self):
        """When since_date is None yfinance should use period='5y'."""
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = _make_history_df()

        with patch(
            "reit_dashboard.data.stock_price_client.yf.Ticker",
            return_value=mock_ticker,
        ):
            client = StockPriceClient()
            client.fetch_weekly_prices("PLD")

        _, kwargs = mock_ticker.history.call_args
        assert kwargs["period"] == "5y"
        assert "start" not in kwargs

    def test_empty_history_returns_empty_list(self):
        """An empty DataFrame from yfinance returns an empty list."""
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = pd.DataFrame()

        with patch(
            "reit_dashboard.data.stock_price_client.yf.Ticker",
            return_value=mock_ticker,
        ):
            client = StockPriceClient()
            entries = client.fetch_weekly_prices("UNKNOWN")

        assert entries == []

    def test_yfinance_exception_returns_empty_list(self):
        """An exception from yfinance is swallowed and returns []."""
        with patch(
            "reit_dashboard.data.stock_price_client.yf.Ticker",
            side_effect=Exception("network error"),
        ):
            client = StockPriceClient()
            entries = client.fetch_weekly_prices("PLD")

        assert entries == []

    def test_rate_limit_sleep_called_once(self, monkeypatch):
        """time.sleep must be called exactly once per fetch call."""
        sleep_calls: list[float] = []
        monkeypatch.setattr(
            "reit_dashboard.data.stock_price_client.time.sleep",
            lambda s: sleep_calls.append(s),
        )
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = _make_history_df()

        with patch(
            "reit_dashboard.data.stock_price_client.yf.Ticker",
            return_value=mock_ticker,
        ):
            client = StockPriceClient(rate_limit_delay=1.0)
            client.fetch_weekly_prices("PLD")

        assert len(sleep_calls) == 1
        assert sleep_calls[0] == 1.0
