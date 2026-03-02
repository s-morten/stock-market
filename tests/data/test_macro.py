"""
Unit tests for FRED macro data: FredClient, MacroFactRepository,
and ingest_macro_data().
"""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from reit_dashboard.data.fred_client import FRED_SERIES, FredClient, MacroObservation
from reit_dashboard.data.ingestion import ingest_macro_data
from reit_dashboard.data.models import MacroFact
from reit_dashboard.data.repository import MacroFactRepository


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_observation(
    series_id: str = "FEDFUNDS",
    obs_date: date = date(2024, 1, 1),
    value: float = 5.33,
) -> MacroObservation:
    meta = FRED_SERIES[series_id]
    return MacroObservation(
        series_id=series_id,
        series_name=meta["name"],
        date=obs_date,
        value=value,
        unit=meta["unit"],
        frequency=meta["frequency"],
    )


def _make_fact(
    series_id: str = "FEDFUNDS",
    obs_date: date = date(2024, 1, 1),
    value: float = 5.33,
) -> MacroFact:
    meta = FRED_SERIES[series_id]
    return MacroFact(
        series_id=series_id,
        series_name=meta["name"],
        date=obs_date,
        value=value,
        unit=meta["unit"],
        frequency=meta["frequency"],
    )


# ---------------------------------------------------------------------------
# FredClient
# ---------------------------------------------------------------------------


class TestFredClient:
    """Tests for FredClient.fetch_series() and fetch_all()."""

    def _make_client_with_mock(self, raw_series: pd.Series) -> FredClient:
        """Create a FredClient whose underlying fredapi.Fred is mocked."""
        with patch("reit_dashboard.data.fred_client.FredClient.__init__", return_value=None):
            client = FredClient.__new__(FredClient)
        mock_fred = MagicMock()
        mock_fred.get_series.return_value = raw_series
        client._fred = mock_fred
        return client

    def test_fetch_series_returns_observations(self):
        """fetch_series converts fredapi rows to MacroObservation instances."""
        dates = pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01"])
        raw = pd.Series([5.33, 5.33, 5.33], index=dates)
        client = self._make_client_with_mock(raw)

        obs = client.fetch_series("FEDFUNDS")

        assert len(obs) == 3
        assert all(isinstance(o, MacroObservation) for o in obs)
        assert obs[0].series_id == "FEDFUNDS"
        assert obs[0].value == pytest.approx(5.33)
        assert obs[0].date == date(2024, 1, 1)

    def test_fetch_series_drops_nan(self):
        """NaN values are skipped during conversion."""
        dates = pd.to_datetime(["2024-01-01", "2024-02-01"])
        raw = pd.Series([float("nan"), 5.25], index=dates)
        client = self._make_client_with_mock(raw)

        obs = client.fetch_series("FEDFUNDS")

        assert len(obs) == 1
        assert obs[0].value == pytest.approx(5.25)

    def test_fetch_series_sorted_ascending(self):
        """Returned observations are sorted by date ascending."""
        dates = pd.to_datetime(["2024-03-01", "2024-01-01", "2024-02-01"])
        raw = pd.Series([3.0, 1.0, 2.0], index=dates)
        client = self._make_client_with_mock(raw)

        obs = client.fetch_series("FEDFUNDS")

        assert [o.value for o in obs] == pytest.approx([1.0, 2.0, 3.0])

    def test_fetch_series_unknown_id_raises(self):
        """Passing an unknown series ID raises ValueError."""
        client = self._make_client_with_mock(pd.Series([], dtype=float))
        with pytest.raises(ValueError, match="Unknown FRED series"):
            client.fetch_series("INVALID_SERIES")

    def test_fetch_all_returns_combined(self):
        """fetch_all() collects observations from all configured series."""
        dates = pd.to_datetime(["2024-01-01"])
        raw = pd.Series([1.0], index=dates)
        client = self._make_client_with_mock(raw)
        # All series use the same mock return value.
        all_obs = client.fetch_all()

        expected_count = len(FRED_SERIES) * 1  # 1 obs per series
        assert len(all_obs) == expected_count

    def test_fetch_all_skips_errors(self):
        """fetch_all() logs a warning and skips series that raise."""
        with patch("reit_dashboard.data.fred_client.FredClient.__init__", return_value=None):
            client = FredClient.__new__(FredClient)
        mock_fred = MagicMock()
        mock_fred.get_series.side_effect = RuntimeError("network error")
        client._fred = mock_fred

        all_obs = client.fetch_all()

        assert all_obs == []


# ---------------------------------------------------------------------------
# MacroFactRepository
# ---------------------------------------------------------------------------


class TestMacroFactRepository:
    """Tests for MacroFactRepository CRUD operations."""

    def test_upsert_and_get_series(self, db_session):
        """Upserted facts can be retrieved by series_id."""
        repo = MacroFactRepository(db_session)
        repo.upsert(_make_fact("FEDFUNDS", date(2024, 1, 1), 5.33))
        repo.upsert(_make_fact("FEDFUNDS", date(2024, 2, 1), 5.33))
        repo.upsert(_make_fact("GS10", date(2024, 1, 1), 4.0))
        db_session.flush()

        fedfunds = repo.get_series("FEDFUNDS")
        assert len(fedfunds) == 2
        assert all(f.series_id == "FEDFUNDS" for f in fedfunds)

    def test_upsert_updates_existing(self, db_session):
        """Upserting the same (series_id, date) overwrites the previous value."""
        repo = MacroFactRepository(db_session)
        repo.upsert(_make_fact("FEDFUNDS", date(2024, 1, 1), 5.00))
        repo.upsert(_make_fact("FEDFUNDS", date(2024, 1, 1), 5.33))
        db_session.flush()

        facts = repo.get_series("FEDFUNDS")
        assert len(facts) == 1
        assert float(facts[0].value) == pytest.approx(5.33)

    def test_get_series_date_filter(self, db_session):
        """start_date / end_date filtering works correctly."""
        repo = MacroFactRepository(db_session)
        for month in range(1, 6):
            repo.upsert(_make_fact("GS10", date(2024, month, 1), float(month)))
        db_session.flush()

        subset = repo.get_series("GS10", start_date=date(2024, 2, 1), end_date=date(2024, 4, 1))
        assert len(subset) == 3
        assert [float(f.value) for f in subset] == pytest.approx([2.0, 3.0, 4.0])

    def test_get_all_series_empty(self, db_session):
        """Returns empty list when no macro data exists."""
        repo = MacroFactRepository(db_session)
        assert repo.get_all_series() == []

    def test_get_all_series_multiple(self, db_session):
        """get_all_series returns facts for every series sorted by series_id then date."""
        repo = MacroFactRepository(db_session)
        repo.upsert(_make_fact("GS10", date(2024, 2, 1), 4.0))
        repo.upsert(_make_fact("FEDFUNDS", date(2024, 1, 1), 5.33))
        db_session.flush()

        facts = repo.get_all_series()
        assert len(facts) == 2
        # Sorted by series_id ascending: FEDFUNDS < GS10
        assert facts[0].series_id == "FEDFUNDS"
        assert facts[1].series_id == "GS10"


# ---------------------------------------------------------------------------
# ingest_macro_data
# ---------------------------------------------------------------------------


class TestIngestMacroData:
    """Tests for the ingest_macro_data() orchestration function."""

    def _mock_fred_client(self, observations: list[MacroObservation]) -> MagicMock:
        client = MagicMock(spec=FredClient)
        # Group observations by series so fetch_series returns the right ones.
        by_series: dict[str, list[MacroObservation]] = {}
        for obs in observations:
            by_series.setdefault(obs.series_id, []).append(obs)
        client.fetch_series.side_effect = lambda series_id, **_kw: by_series.get(series_id, [])
        return client

    def test_ingest_persists_observations(self, db_session):
        """ingest_macro_data upserts all returned observations."""
        obs = [
            _make_observation("FEDFUNDS", date(2024, 1, 1), 5.33),
            _make_observation("GS10", date(2024, 1, 1), 4.1),
        ]
        client = self._mock_fred_client(obs)

        result = ingest_macro_data(client, db_session)
        db_session.flush()

        assert result["observations_upserted"] == 2
        repo = MacroFactRepository(db_session)
        assert len(repo.get_all_series()) == 2

    def test_ingest_returns_summary(self, db_session):
        """Return dict contains expected keys."""
        client = self._mock_fred_client([])
        result = ingest_macro_data(client, db_session)
        assert "series_fetched" in result
        assert "observations_upserted" in result
        assert "errors" in result

    def test_ingest_empty_returns_zero(self, db_session):
        """When FRED client returns no data, observations_upserted is 0."""
        client = self._mock_fred_client([])
        result = ingest_macro_data(client, db_session)
        assert result["observations_upserted"] == 0
