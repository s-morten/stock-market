"""
FRED API client for fetching macroeconomic time-series data.

Uses the ``fredapi`` library to access the Federal Reserve Bank of
St. Louis FRED database.  A free API key is required and can be obtained
at https://fred.stlouisfed.org/docs/api/api_key.html

FRED series fetched
-------------------
FEDFUNDS    Federal Funds Effective Rate            (Monthly, %)
GS10        10-Year Treasury Constant Maturity Rate (Monthly, %)
GS2         2-Year Treasury Constant Maturity Rate  (Monthly, %)
UNRATE      Unemployment Rate                       (Monthly, %)
CPILFESL    CPI Less Food & Energy (Core Inflation) (Monthly, index)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

logger = logging.getLogger(__name__)

# FRED series to fetch, with display metadata.
FRED_SERIES: dict[str, dict[str, str]] = {
    "FEDFUNDS": {
        "name": "Federal Funds Rate",
        "unit": "Percent",
        "frequency": "Monthly",
        "category": "policy_rate",
    },
    "GS10": {
        "name": "10-Year Treasury Yield",
        "unit": "Percent",
        "frequency": "Monthly",
        "category": "bond_yield",
    },
    "GS2": {
        "name": "2-Year Treasury Yield",
        "unit": "Percent",
        "frequency": "Monthly",
        "category": "bond_yield",
    },
    "UNRATE": {
        "name": "Unemployment Rate",
        "unit": "Percent",
        "frequency": "Monthly",
        "category": "unemployment",
    },
    "CPILFESL": {
        "name": "Core CPI (Less Food & Energy)",
        "unit": "Index 1982-84=100",
        "frequency": "Monthly",
        "category": "inflation",
    },
}


@dataclass
class MacroObservation:
    """
    A single macroeconomic observation.

    Attributes:
        series_id:   FRED series identifier.
        series_name: Human-readable name.
        date:        Observation date.
        value:       Numeric value.
        unit:        Unit of measure.
        frequency:   Reporting frequency.
    """

    series_id: str
    series_name: str
    date: date
    value: float
    unit: str
    frequency: str


class FredClient:
    """
    Thin wrapper around the ``fredapi`` library.

    Parameters:
        api_key: FRED API key.
    """

    def __init__(self, api_key: str) -> None:
        from fredapi import Fred  # imported lazily so tests can mock easily

        self._fred = Fred(api_key=api_key)

    def fetch_series(
        self,
        series_id: str,
        since_date: date | None = None,
    ) -> list[MacroObservation]:
        """
        Fetch all observations for a FRED series.

        Parameters:
            series_id:   FRED series identifier (e.g. "FEDFUNDS").
            since_date:  Earliest observation date to include.  Defaults
                         to fetching all available history.

        Returns:
            list[MacroObservation]: Observations sorted by date ascending.

        Raises:
            ValueError: If the series ID is not in :data:`FRED_SERIES`.
            Exception:  On network or FRED API errors.
        """
        meta = FRED_SERIES.get(series_id)
        if meta is None:
            raise ValueError(
                f"Unknown FRED series {series_id!r}. "
                f"Supported: {list(FRED_SERIES)}"
            )

        kwargs: dict[str, Any] = {}
        if since_date is not None:
            kwargs["observation_start"] = since_date.isoformat()

        logger.info("Fetching FRED series %s (%s) …", series_id, meta["name"])
        raw = self._fred.get_series(series_id, **kwargs)

        observations: list[MacroObservation] = []
        for ts, val in raw.items():
            # fredapi returns a pandas Series indexed by Timestamp.
            if val != val:  # NaN guard
                continue
            obs_date = ts.date() if hasattr(ts, "date") else datetime.fromisoformat(str(ts)).date()
            observations.append(
                MacroObservation(
                    series_id=series_id,
                    series_name=meta["name"],
                    date=obs_date,
                    value=float(val),
                    unit=meta["unit"],
                    frequency=meta["frequency"],
                )
            )

        logger.info("  → %d observations for %s", len(observations), series_id)
        return sorted(observations, key=lambda o: o.date)

    def fetch_all(
        self, since_date: date | None = None
    ) -> list[MacroObservation]:
        """
        Fetch all configured FRED series.

        Silently skips any series that raises an error (logs a warning).

        Parameters:
            since_date: Earliest date to include across all series.

        Returns:
            list[MacroObservation]: Combined observations from all series.
        """
        all_obs: list[MacroObservation] = []
        for series_id in FRED_SERIES:
            try:
                all_obs.extend(self.fetch_series(series_id, since_date=since_date))
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to fetch FRED series %s: %s", series_id, exc)
        return all_obs
