"""
Stock price client for fetching weekly OHLCV data via Yahoo Finance.

Uses the ``yfinance`` library which wraps the Yahoo Finance API.
Yahoo Finance does not publish hard rate limits; community experience
suggests staying well under 2 000 requests/hour.  For the PoC (5 tickers)
a 1-second pause between ticker fetches is more than sufficient.

No API key is required.
"""

import time
from datetime import date
from typing import Any

import yfinance as yf
from pydantic import BaseModel

# Minimum seconds to wait between yfinance history() calls.
# 1 s ≈ 60 req/min which is safely below observed Yahoo Finance limits.
_RATE_LIMIT_DELAY_SECONDS: float = 1.0


class StockPriceEntry(BaseModel):
    """
    One week's OHLCV price bar for a single ticker.

    Attributes:
        date:   Week start date of the bar (Monday, typically).
        open:   Opening price (USD, adjusted for splits/dividends).
        high:   Intra-week high (USD, adjusted).
        low:    Intra-week low (USD, adjusted).
        close:  Adjusted closing price (USD).
        volume: Total shares traded during the week.
    """

    date: date
    open: float
    high: float
    low: float
    close: float
    volume: int


class StockPriceClient:
    """
    Thin wrapper around yfinance for fetching weekly stock price history.

    Parameters:
        rate_limit_delay: Seconds to sleep before each API call (default 1.0).
    """

    def __init__(self, rate_limit_delay: float = _RATE_LIMIT_DELAY_SECONDS) -> None:
        self._delay = rate_limit_delay

    def fetch_weekly_prices(
        self,
        ticker: str,
        since_date: date | None = None,
    ) -> list[StockPriceEntry]:
        """
        Fetch weekly adjusted OHLCV prices for a single ticker.

        A rate-limit delay is observed *before* the network call so that
        callers can invoke this method in a tight loop without exceeding
        Yahoo Finance's request rate.

        Parameters:
            ticker:     Exchange ticker symbol (e.g. "PLD").
            since_date: Earliest date to include.  When None the last 5
                        years are fetched using yfinance's "5y" period.

        Returns:
            list[StockPriceEntry]: Weekly bars sorted by date ascending.
                Returns an empty list if the ticker is unknown or the
                request fails.
        """
        time.sleep(self._delay)

        try:
            t = yf.Ticker(ticker)
            kwargs: dict[str, Any] = {"interval": "1wk", "auto_adjust": True}
            if since_date is not None:
                kwargs["start"] = since_date.isoformat()
            else:
                kwargs["period"] = "5y"

            hist = t.history(**kwargs)
        except Exception:  # noqa: BLE001
            # yfinance can raise various errors for unknown tickers or
            # network issues; return empty list so ingestion can continue.
            return []

        if hist.empty:
            return []

        entries: list[StockPriceEntry] = []
        for ts, row in hist.iterrows():
            entries.append(
                StockPriceEntry(
                    date=ts.date(),
                    open=float(row.get("Open", 0.0)),
                    high=float(row.get("High", 0.0)),
                    low=float(row.get("Low", 0.0)),
                    close=float(row.get("Close", 0.0)),
                    volume=int(row.get("Volume", 0)),
                )
            )

        return sorted(entries, key=lambda e: e.date)
