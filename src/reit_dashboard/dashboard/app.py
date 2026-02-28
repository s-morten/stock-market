"""
Streamlit dashboard for REIT financial data analysis.

Run with:
    uv run streamlit run src/reit_dashboard/dashboard/app.py

Reads data from the local database (configured via DATABASE_URL env var).
Assumes the database has been seeded via scripts/ingest_poc.py.
"""

from datetime import date, timedelta
from decimal import Decimal

import altair as alt
import pandas as pd
import streamlit as st
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from reit_dashboard.config import get_database_url
from reit_dashboard.data.models import Base
from reit_dashboard.data.repository import (
    CompanyRepository,
    FinancialFactRepository,
    StockPriceRepository,
)

# ---------------------------------------------------------------------------
# Database helpers (cached so the connection is reused across Streamlit runs)
# ---------------------------------------------------------------------------


@st.cache_resource
def get_session_factory():
    """
    Create and cache a SQLAlchemy session factory.

    Returns:
        sessionmaker: Bound session factory.
    """
    engine = create_engine(get_database_url(), echo=False)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------


def load_companies(session_factory) -> pd.DataFrame:
    """
    Load all companies from the database into a DataFrame.

    Parameters:
        session_factory: SQLAlchemy sessionmaker.

    Returns:
        pd.DataFrame: Columns cik, name, sic, fiscal_year_end, ticker.
    """
    with session_factory() as session:
        repo = CompanyRepository(session)
        companies = repo.list_all()
    return pd.DataFrame(
        [
            {
                "cik": c.cik,
                "name": c.name,
                "sic": c.sic,
                "fiscal_year_end": c.fiscal_year_end,
                "ticker": c.ticker,
            }
            for c in companies
        ]
    )


def load_facts(
    session_factory,
    ciks: list[str],
    concept: str,
    form: str,
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    """
    Load financial facts for the selected companies and filters.

    Adds a ``quarter_label`` column (e.g. "2022-Q3") derived from the
    period end date for use in quarterly charts.

    Parameters:
        session_factory: SQLAlchemy sessionmaker.
        ciks:            List of CIKs to include.
        concept:         XBRL concept name.
        form:            Filing form type.
        start_date:      Earliest period_end to include.
        end_date:        Latest period_end to include.

    Returns:
        pd.DataFrame: Columns cik, company_name, concept, period_end,
            quarter_label, value, unit.
    """
    rows: list[dict] = []
    with session_factory() as session:
        fact_repo = FinancialFactRepository(session)
        company_repo = CompanyRepository(session)
        for cik in ciks:
            company = company_repo.get_by_cik(cik)
            company_name = company.name if company else cik
            facts = fact_repo.get_facts(
                cik=cik,
                concept=concept,
                form=form,
                start_date=start_date,
                end_date=end_date,
            )
            for f in facts:
                rows.append(
                    {
                        "cik": f.cik,
                        "company_name": company_name,
                        "concept": f.concept,
                        "period_end": f.period_end,
                        "value": float(f.value),
                        "unit": f.unit,
                    }
                )

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["period_end"] = pd.to_datetime(df["period_end"])
    # Quarter label: "YYYY-Qn" for axis readability.
    df["quarter_label"] = (
        df["period_end"].dt.year.astype(str)
        + "-Q"
        + df["period_end"].dt.quarter.astype(str)
    )
    return df


def load_stock_prices(
    session_factory,
    tickers: list[str],
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    """
    Load weekly stock prices for the given tickers from the database.

    Parameters:
        session_factory: SQLAlchemy sessionmaker.
        tickers:         List of exchange ticker symbols.
        start_date:      Earliest date to include.
        end_date:        Latest date to include.

    Returns:
        pd.DataFrame: Columns ticker, date, open, high, low, close, volume.
                      Returns empty DataFrame when no data is available.
    """
    if not tickers:
        return pd.DataFrame()

    rows: list[dict] = []
    with session_factory() as session:
        repo = StockPriceRepository(session)
        prices = repo.get_prices_for_tickers(
            tickers, start_date=start_date, end_date=end_date
        )
        for p in prices:
            rows.append(
                {
                    "ticker": p.ticker,
                    "date": p.date,
                    "open": float(p.open),
                    "high": float(p.high),
                    "low": float(p.low),
                    "close": float(p.close),
                    "volume": p.volume,
                }
            )

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df


# ---------------------------------------------------------------------------
# Chart helpers
# ---------------------------------------------------------------------------

_CONCEPT_LABELS: dict[str, str] = {
    "Revenues": "Revenue (USD)",
    "NetIncomeLoss": "Net Income / Loss (USD)",
    "Assets": "Total Assets (USD)",
    "Liabilities": "Total Liabilities (USD)",
}


def build_trend_chart(df: pd.DataFrame, concept: str) -> alt.Chart:
    """
    Build an Altair multi-line trend chart over time.

    Parameters:
        df:      DataFrame with company_name, period_end, value columns.
        concept: XBRL concept name used for axis labels.

    Returns:
        alt.Chart: Interactive line + point chart.
    """
    y_label = _CONCEPT_LABELS.get(concept, f"{concept} (USD)")
    return (
        alt.Chart(df)
        .mark_line(point=True)
        .encode(
            x=alt.X(
                "period_end:T",
                title="Period End",
                axis=alt.Axis(format="%b %Y", labelAngle=-45),
            ),
            y=alt.Y("value:Q", title=y_label, axis=alt.Axis(format="~s")),
            color=alt.Color("company_name:N", title="Company"),
            tooltip=[
                alt.Tooltip("company_name:N", title="Company"),
                alt.Tooltip("quarter_label:N", title="Quarter"),
                alt.Tooltip("value:Q", title="Value (USD)", format=",.0f"),
            ],
        )
        .properties(
            title=f"{concept} – quarterly trend",
            height=380,
        )
        .interactive()
    )


def build_bar_chart(df: pd.DataFrame, concept: str) -> alt.Chart:
    """
    Build a grouped bar chart comparing companies per quarter.

    Parameters:
        df:      DataFrame with company_name, quarter_label, value columns.
        concept: XBRL concept name used for axis labels.

    Returns:
        alt.Chart: Grouped bar chart.
    """
    y_label = _CONCEPT_LABELS.get(concept, f"{concept} (USD)")
    return (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X(
                "quarter_label:O",
                title="Quarter",
                sort=sorted(df["quarter_label"].unique()),
                axis=alt.Axis(labelAngle=-45),
            ),
            y=alt.Y("value:Q", title=y_label, axis=alt.Axis(format="~s")),
            color=alt.Color("company_name:N", title="Company"),
            xOffset="company_name:N",
            tooltip=[
                alt.Tooltip("company_name:N", title="Company"),
                alt.Tooltip("quarter_label:N", title="Quarter"),
                alt.Tooltip("value:Q", title="Value (USD)", format=",.0f"),
            ],
        )
        .properties(
            title=f"{concept} – quarterly comparison",
            height=380,
        )
    )

def build_stock_price_chart(
    price_df: pd.DataFrame,
    ticker_to_name: dict[str, str],
) -> alt.Chart:
    """
    Build an Altair line chart for weekly adjusted close prices.

    Parameters:
        price_df:       DataFrame with columns ticker, date, close.
        ticker_to_name: Mapping from ticker symbol to company name.

    Returns:
        alt.Chart: Interactive multi-line price chart.
    """
    df = price_df.copy()
    df["company_name"] = df["ticker"].map(
        lambda t: ticker_to_name.get(t, t)
    )
    return (
        alt.Chart(df)
        .mark_line(point=False)
        .encode(
            x=alt.X(
                "date:T",
                title="Week",
                axis=alt.Axis(format="%b %Y", labelAngle=-45),
            ),
            y=alt.Y(
                "close:Q",
                title="Adjusted Close (USD)",
                axis=alt.Axis(format="$.2f"),
            ),
            color=alt.Color("company_name:N", title="Company"),
            tooltip=[
                alt.Tooltip("company_name:N", title="Company"),
                alt.Tooltip("ticker:N", title="Ticker"),
                alt.Tooltip("date:T", title="Week", format="%Y-%m-%d"),
                alt.Tooltip("close:Q", title="Close (USD)", format="$.2f"),
                alt.Tooltip(
                    "volume:Q", title="Volume", format=",d"
                ),
            ],
        )
        .properties(
            title="Weekly adjusted close price – last 5 years",
            height=380,
        )
        .interactive()
    )


def build_normalised_price_chart(
    price_df: pd.DataFrame,
    ticker_to_name: dict[str, str],
) -> alt.Chart:
    """
    Build a base-100 normalised price chart for relative comparison.

    Re-bases each series to 100 at the earliest available date so
    companies with very different price levels can be compared directly.

    Parameters:
        price_df:       DataFrame with columns ticker, date, close.
        ticker_to_name: Mapping from ticker symbol to company name.

    Returns:
        alt.Chart: Interactive normalised line chart.
    """
    df = price_df.copy()
    df["company_name"] = df["ticker"].map(
        lambda t: ticker_to_name.get(t, t)
    )
    # Compute base-100 index per ticker.
    first_close = (
        df.sort_values("date")
        .groupby("ticker")["close"]
        .first()
        .rename("first_close")
    )
    df = df.merge(first_close, on="ticker")
    df["indexed"] = df["close"] / df["first_close"] * 100

    return (
        alt.Chart(df)
        .mark_line()
        .encode(
            x=alt.X(
                "date:T",
                title="Week",
                axis=alt.Axis(format="%b %Y", labelAngle=-45),
            ),
            y=alt.Y(
                "indexed:Q",
                title="Relative Price (base = 100)",
                axis=alt.Axis(format=".1f"),
            ),
            color=alt.Color("company_name:N", title="Company"),
            tooltip=[
                alt.Tooltip("company_name:N", title="Company"),
                alt.Tooltip("date:T", title="Week", format="%Y-%m-%d"),
                alt.Tooltip(
                    "indexed:Q", title="Indexed Value", format=".2f"
                ),
                alt.Tooltip("close:Q", title="Close (USD)", format="$.2f"),
            ],
        )
        .properties(
            title="Relative price performance (base 100)",
            height=380,
        )
        .interactive()
    )


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------


def render_metrics(df: pd.DataFrame, companies_df: pd.DataFrame) -> None:
    """
    Render a row of Streamlit metric cards (latest value + YoY change).

    Shows one card per selected company with the most recent reported
    value and the year-over-year change (same quarter, prior year).

    Parameters:
        df:           Facts DataFrame (already filtered by concept/form).
        companies_df: Company metadata DataFrame.
    """
    cik_to_name = companies_df.set_index("cik")["name"].to_dict()
    cols = st.columns(len(df["cik"].unique()))

    for col, cik in zip(cols, sorted(df["cik"].unique())):
        company_df = df[df["cik"] == cik].sort_values("period_end")
        if company_df.empty:
            continue

        latest = company_df.iloc[-1]
        latest_val = latest["value"]
        latest_quarter = latest["quarter_label"]

        # Find the same quarter one year prior for YoY delta.
        one_year_ago = latest["period_end"] - pd.DateOffset(years=1)
        prior = company_df[
            (company_df["period_end"] >= one_year_ago - pd.Timedelta(days=45))
            & (company_df["period_end"] <= one_year_ago + pd.Timedelta(days=45))
        ]
        delta_str: str | None = None
        if not prior.empty:
            prior_val = prior.iloc[-1]["value"]
            if prior_val != 0:
                pct = (latest_val - prior_val) / abs(prior_val) * 100
                delta_str = f"{pct:+.1f}% YoY"

        col.metric(
            label=cik_to_name.get(cik, cik),
            value=f"${latest_val / 1e6:,.1f}M",
            delta=delta_str,
            help=f"Latest: {latest_quarter}",
        )


# ---------------------------------------------------------------------------
# Main Streamlit app
# ---------------------------------------------------------------------------


def main() -> None:
    """Render the REIT Dashboard Streamlit application."""
    st.set_page_config(
        page_title="REIT Dashboard",
        page_icon="🏢",
        layout="wide",
    )
    st.title("🏢 US REIT Financial Dashboard")
    st.caption(
        "Data sourced from SEC EDGAR · Quarterly (10-Q) · Last 5 years"
    )

    session_factory = get_session_factory()
    companies_df = load_companies(session_factory)

    if companies_df.empty:
        st.warning(
            "No data found. Run `uv run python scripts/ingest_poc.py` "
            "first to populate the database."
        )
        return

    # --- Sidebar filters ---
    st.sidebar.header("Filters")

    company_options = companies_df.set_index("cik")["name"].to_dict()
    selected_ciks = st.sidebar.multiselect(
        "Companies",
        options=list(company_options.keys()),
        default=list(company_options.keys()),
        format_func=lambda cik: company_options[cik],
    )

    concept = st.sidebar.selectbox(
        "Financial Concept",
        options=["Revenues", "NetIncomeLoss", "Assets", "Liabilities"],
        format_func=lambda c: _CONCEPT_LABELS.get(c, c),
    )

    # Default to 10-Q since ingestion focuses on quarterly data.
    form = st.sidebar.radio("Filing Type", options=["10-Q", "10-K"], index=0)

    default_start = date.today() - timedelta(days=365 * 5)
    start_date = st.sidebar.date_input("Start Date", value=default_start)
    end_date = st.sidebar.date_input("End Date", value=date.today())

    # --- Company info table ---
    st.subheader("Company Overview")
    display_companies = companies_df[companies_df["cik"].isin(selected_ciks)]
    st.dataframe(
        display_companies.rename(columns={
            "cik": "CIK",
            "name": "Name",
            "sic": "SIC",
            "fiscal_year_end": "FY End",
            "ticker": "Ticker",
        }),
        use_container_width=True,
        hide_index=True,
    )

    if not selected_ciks:
        st.info("Select at least one company in the sidebar.")
        return

    # Build ticker → name mapping for selected companies.
    ticker_to_name: dict[str, str] = {}
    selected_tickers: list[str] = []
    for _, row in display_companies.iterrows():
        if row.get("ticker"):
            ticker_to_name[row["ticker"]] = row["name"]
            selected_tickers.append(row["ticker"])

    # --- Financial facts ---
    facts_df = load_facts(
        session_factory, selected_ciks, concept, form, start_date, end_date
    )

    # --- Stock prices ---
    price_df = load_stock_prices(
        session_factory, selected_tickers, start_date, end_date
    )

    # ================================================================== #
    # Section 1 – Financial Fundamentals                                  #
    # ================================================================== #
    st.header("📊 Financial Fundamentals")

    if facts_df.empty:
        st.warning(
            f"No **{form}** data found for **{concept}** in the selected "
            "period. Try adjusting the filters or re-running ingestion."
        )
    else:
        # Metrics row
        st.subheader(f"Latest {_CONCEPT_LABELS.get(concept, concept)}")
        render_metrics(facts_df, companies_df)

        st.divider()

        tab_trend, tab_bar = st.tabs(["📈 Trend", "📊 Quarter Comparison"])
        with tab_trend:
            st.altair_chart(
                build_trend_chart(facts_df, concept),
                use_container_width=True,
            )
        with tab_bar:
            st.altair_chart(
                build_bar_chart(facts_df, concept),
                use_container_width=True,
            )

        with st.expander("Raw Financial Data"):
            st.dataframe(
                facts_df.rename(columns={
                    "cik": "CIK",
                    "company_name": "Company",
                    "concept": "Concept",
                    "period_end": "Period End",
                    "quarter_label": "Quarter",
                    "value": "Value (USD)",
                    "unit": "Unit",
                })[["CIK", "Company", "Concept", "Quarter",
                    "Period End", "Value (USD)", "Unit"]],
                use_container_width=True,
                hide_index=True,
            )

    # ================================================================== #
    # Section 2 – Stock Prices                                           #
    # ================================================================== #
    st.header("💹 Stock Prices")

    if price_df.empty:
        st.warning(
            "No stock price data found for the selected period. "
            "Run `uv run python scripts/ingest_poc.py` to populate prices."
        )
    else:
        # Latest price metrics
        price_cols = st.columns(len(price_df["ticker"].unique()))
        for col, ticker in zip(
            price_cols, sorted(price_df["ticker"].unique())
        ):
            t_df = price_df[price_df["ticker"] == ticker].sort_values("date")
            latest_price = t_df.iloc[-1]["close"]
            # WoW change
            delta_str: str | None = None
            if len(t_df) >= 2:
                prior_price = t_df.iloc[-2]["close"]
                if prior_price != 0:
                    wow = (latest_price - prior_price) / prior_price * 100
                    delta_str = f"{wow:+.2f}% WoW"
            col.metric(
                label=f"{ticker_to_name.get(ticker, ticker)} ({ticker})",
                value=f"${latest_price:,.2f}",
                delta=delta_str,
                help=f"Week of {t_df.iloc[-1]['date'].date()}",
            )

        st.divider()

        tab_abs, tab_norm = st.tabs(
            ["📈 Price History", "📊 Relative Performance"]
        )
        with tab_abs:
            st.altair_chart(
                build_stock_price_chart(price_df, ticker_to_name),
                use_container_width=True,
            )
        with tab_norm:
            st.altair_chart(
                build_normalised_price_chart(price_df, ticker_to_name),
                use_container_width=True,
            )

        with st.expander("Raw Price Data"):
            st.dataframe(
                price_df.rename(columns={
                    "ticker": "Ticker",
                    "date": "Week",
                    "open": "Open",
                    "high": "High",
                    "low": "Low",
                    "close": "Close (adj.)",
                    "volume": "Volume",
                })[["Ticker", "Week", "Open", "High",
                    "Low", "Close (adj.)", "Volume"]],
                use_container_width=True,
                hide_index=True,
            )


if __name__ == "__main__":
    main()
