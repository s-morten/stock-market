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
    PropertyFactRepository,
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


def load_property_facts(
    session_factory,
    ciks: list[str],
    metric_name: str | None = None,
    form: str | None = None,
) -> pd.DataFrame:
    """
    Load parsed Item 2 property facts for the selected companies.

    Parameters:
        session_factory: SQLAlchemy sessionmaker.
        ciks:            List of CIKs to include.
        metric_name:     Optional filter on the normalised metric key.
        form:            Optional form type filter (e.g. "10-K").

    Returns:
        pd.DataFrame: Columns cik, ticker, company_name, period_end,
            form, accn, property_type, metric_name, value_numeric.
            Returns empty DataFrame when no data is available.
    """
    if not ciks:
        return pd.DataFrame()

    rows: list[dict] = []
    with session_factory() as session:
        prop_repo = PropertyFactRepository(session)
        company_repo = CompanyRepository(session)
        name_cache: dict[str, str] = {}

        facts = prop_repo.get_facts_for_ciks(ciks, metric_name=metric_name, form=form)
        for f in facts:
            if f.cik not in name_cache:
                company = company_repo.get_by_cik(f.cik)
                name_cache[f.cik] = company.name if company else f.cik
            # Attempt numeric conversion; keep NaN for non-numeric values.
            try:
                numeric = float(f.value.strip("%"))
                if "%" in f.value:
                    numeric = numeric  # keep as-is; caller decides to /100
            except (ValueError, AttributeError):
                numeric = float("nan")
            rows.append(
                {
                    "cik": f.cik,
                    "ticker": f.ticker,
                    "company_name": name_cache[f.cik],
                    "period_end": f.period_end,
                    "form": f.form,
                    "accn": f.accn,
                    "property_type": f.property_type,
                    "metric_name": f.metric_name,
                    "value_str": f.value,
                    "value_numeric": numeric,
                }
            )

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["period_end"] = pd.to_datetime(df["period_end"], errors="coerce")
    return df




_CONCEPT_LABELS: dict[str, str] = {
    "Revenues": "Revenue (USD)",
    "NetIncomeLoss": "Net Income / Loss (USD)",
    "Assets": "Total Assets (USD)",
    "Liabilities": "Total Liabilities (USD)",
    "LongTermDebt": "Long-Term Debt (USD)",
    "LongTermDebtNoncurrent": "Long-Term Debt Non-current (USD)",
    "ShortTermBorrowings": "Short-Term Borrowings (USD)",
    "InterestExpense": "Interest Expense (USD)",
    "InterestAndDebtExpense": "Interest & Debt Expense (USD)",
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
        options=[
            "Revenues", "NetIncomeLoss", "Assets", "Liabilities",
            "LongTermDebt", "LongTermDebtNoncurrent",
            "ShortTermBorrowings", "InterestExpense", "InterestAndDebtExpense",
        ],
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

    # ================================================================== #
    # Section 3 – Property Portfolio                                      #
    # ================================================================== #
    st.header("🏗️ Property Portfolio")
    st.caption(
        "Total property counts are extracted by Gemini from Item 2 of 10-K filings. "
        "Run `uv run python scripts/ingest_poc.py` (with `GEMINI_API_KEY` set) to populate."
    )

    prop_df_all = load_property_facts(session_factory, selected_ciks)

    # ------------------------------------------------------------------
    # 3a – Gemini-extracted total property count (time-series)
    # ------------------------------------------------------------------
    gemini_df = pd.DataFrame()
    if not prop_df_all.empty:
        gemini_df = prop_df_all[
            prop_df_all["metric_name"] == "total_properties_gemini"
        ].copy()
        gemini_df = gemini_df.dropna(subset=["value_numeric"])

    if gemini_df.empty:
        st.info(
            "No Gemini property count data found. "
            "Re-run ingestion with `GEMINI_API_KEY` set to populate."
        )
    else:
        count_chart = (
            alt.Chart(gemini_df)
            .mark_line(point=True)
            .encode(
                x=alt.X("period_end:T", title="Period End",
                        axis=alt.Axis(format="%b %Y", labelAngle=-45)),
                y=alt.Y("value_numeric:Q", title="Total Properties",
                        axis=alt.Axis(format=",.0f"), scale=alt.Scale(zero=False)),
                color=alt.Color("company_name:N", title="Company"),
                tooltip=[
                    alt.Tooltip("company_name:N", title="Company"),
                    alt.Tooltip("period_end:T", title="Period", format="%Y-%m-%d"),
                    alt.Tooltip("value_numeric:Q", title="Total Properties",
                                format=",.0f"),
                ],
            )
            .properties(
                title="Total Properties per Company (Gemini-extracted from 10-K Item 2)",
                height=400,
            )
            .interactive()
        )
        st.altair_chart(count_chart, use_container_width=True)

        # Latest snapshot table
        latest_snap = (
            gemini_df.sort_values("period_end")
            .groupby("company_name")
            .last()
            .reset_index()[["company_name", "ticker", "period_end", "value_numeric"]]
            .rename(columns={
                "company_name": "Company",
                "ticker": "Ticker",
                "period_end": "Latest Filing",
                "value_numeric": "Total Properties",
            })
        )
        latest_snap["Total Properties"] = latest_snap["Total Properties"].astype(int)
        st.dataframe(latest_snap, use_container_width=True, hide_index=True)

    st.divider()

    # ------------------------------------------------------------------
    # 3b – Raw Item 2 HTML-parsed data (detailed breakdown)
    # ------------------------------------------------------------------
    with st.expander("📋 Detailed Item 2 Table Data (HTML-parsed, for reference)", expanded=False):
        raw_df = prop_df_all[
            prop_df_all["metric_name"] != "total_properties_gemini"
        ] if not prop_df_all.empty else prop_df_all

        if raw_df.empty:
            st.info("No detailed breakdown data. Re-run ingestion to populate.")
        else:
            st.dataframe(
                raw_df.rename(columns={
                    "cik": "CIK",
                    "ticker": "Ticker",
                    "company_name": "Company",
                    "period_end": "Period End",
                    "form": "Form",
                    "property_type": "Property Type",
                    "metric_name": "Metric",
                    "value_str": "Value (raw)",
                    "value_numeric": "Value (numeric)",
                })[["Company", "Ticker", "Period End", "Form",
                    "Property Type", "Metric", "Value (raw)", "Value (numeric)"]],
                use_container_width=True,
                hide_index=True,
            )
    st.divider()

    # ================================================================== #
    # Section 4 – Debt & Interest                                         #
    # ================================================================== #
    st.header("🏦 Debt & Interest")
    st.caption(
        "Long-term debt, short-term borrowings, and interest expense from "
        "SEC EDGAR XBRL (10-Q quarterly filings)."
    )

    # Load debt and interest facts for selected companies/period.
    _DEBT_CONCEPTS = [
        "LongTermDebt",
        "LongTermDebtNoncurrent",
        "ShortTermBorrowings",
        "InterestExpense",
        "InterestAndDebtExpense",
    ]

    debt_frames: list[pd.DataFrame] = []
    for _concept in _DEBT_CONCEPTS:
        _df = load_facts(
            session_factory, selected_ciks, _concept, "10-Q", start_date, end_date
        )
        if not _df.empty:
            debt_frames.append(_df)

    debt_df = pd.concat(debt_frames, ignore_index=True) if debt_frames else pd.DataFrame()

    if debt_df.empty:
        st.info(
            "No debt/interest data found. Re-run ingestion to populate. "
            "New debt concepts (LongTermDebt, InterestExpense, etc.) are "
            "fetched automatically on the next ingestion run."
        )
    else:
        # Split into debt vs interest for separate charts.
        _DEBT_ONLY = {"LongTermDebt", "LongTermDebtNoncurrent", "ShortTermBorrowings"}
        _INT_ONLY = {"InterestExpense", "InterestAndDebtExpense"}

        debt_only_df = debt_df[debt_df["concept"].isin(_DEBT_ONLY)]
        int_only_df = debt_df[debt_df["concept"].isin(_INT_ONLY)]

        tab_debt, tab_interest, tab_ratio = st.tabs(
            ["📉 Debt Over Time", "💸 Interest Expense", "⚖️ Debt-to-Assets"]
        )

        with tab_debt:
            if debt_only_df.empty:
                st.info("No debt data available.")
            else:
                debt_chart = (
                    alt.Chart(debt_only_df)
                    .mark_line(point=True)
                    .encode(
                        x=alt.X("period_end:T", title="Period End",
                                axis=alt.Axis(format="%b %Y", labelAngle=-45)),
                        y=alt.Y("value:Q", title="USD",
                                axis=alt.Axis(format="~s"), scale=alt.Scale(zero=True)),
                        color=alt.Color("company_name:N", title="Company"),
                        strokeDash=alt.StrokeDash("concept:N", title="Concept"),
                        tooltip=[
                            alt.Tooltip("company_name:N", title="Company"),
                            alt.Tooltip("concept:N", title="Concept"),
                            alt.Tooltip("period_end:T", title="Period", format="%Y-%m-%d"),
                            alt.Tooltip("value:Q", title="USD", format="$,.0f"),
                        ],
                    )
                    .properties(title="Long-Term & Short-Term Debt", height=400)
                    .interactive()
                )
                st.altair_chart(debt_chart, use_container_width=True)

        with tab_interest:
            if int_only_df.empty:
                st.info("No interest expense data available.")
            else:
                int_chart = (
                    alt.Chart(int_only_df)
                    .mark_bar()
                    .encode(
                        x=alt.X("period_end:T", title="Period End",
                                axis=alt.Axis(format="%b %Y", labelAngle=-45)),
                        y=alt.Y("value:Q", title="USD",
                                axis=alt.Axis(format="~s")),
                        color=alt.Color("company_name:N", title="Company"),
                        xOffset="company_name:N",
                        tooltip=[
                            alt.Tooltip("company_name:N", title="Company"),
                            alt.Tooltip("concept:N", title="Concept"),
                            alt.Tooltip("period_end:T", title="Period", format="%Y-%m-%d"),
                            alt.Tooltip("value:Q", title="USD", format="$,.0f"),
                        ],
                    )
                    .properties(title="Interest & Debt Expense per Quarter", height=400)
                    .interactive()
                )
                st.altair_chart(int_chart, use_container_width=True)

        with tab_ratio:
            # Debt-to-Assets = LongTermDebt / Assets (per company, per period).
            assets_df = load_facts(
                session_factory, selected_ciks, "Assets", "10-Q", start_date, end_date
            )
            ltd_df = debt_only_df[debt_only_df["concept"] == "LongTermDebt"]

            if assets_df.empty or ltd_df.empty:
                st.info("Need both Assets and LongTermDebt data to compute ratio.")
            else:
                merged = pd.merge(
                    ltd_df[["cik", "company_name", "period_end", "value"]].rename(
                        columns={"value": "ltd"}
                    ),
                    assets_df[["cik", "period_end", "value"]].rename(
                        columns={"value": "assets"}
                    ),
                    on=["cik", "period_end"],
                    how="inner",
                )
                merged = merged[merged["assets"] > 0].copy()
                merged["ratio"] = merged["ltd"] / merged["assets"]

                ratio_chart = (
                    alt.Chart(merged)
                    .mark_line(point=True)
                    .encode(
                        x=alt.X("period_end:T", title="Period End",
                                axis=alt.Axis(format="%b %Y", labelAngle=-45)),
                        y=alt.Y("ratio:Q", title="LTD / Assets",
                                axis=alt.Axis(format=".0%"),
                                scale=alt.Scale(zero=True)),
                        color=alt.Color("company_name:N", title="Company"),
                        tooltip=[
                            alt.Tooltip("company_name:N", title="Company"),
                            alt.Tooltip("period_end:T", title="Period", format="%Y-%m-%d"),
                            alt.Tooltip("ratio:Q", title="LTD/Assets", format=".1%"),
                        ],
                    )
                    .properties(title="Debt-to-Assets Ratio (LTD / Total Assets)", height=400)
                    .interactive()
                )
                st.altair_chart(ratio_chart, use_container_width=True)

        with st.expander("Raw Debt & Interest Data", expanded=False):
            st.dataframe(
                debt_df.rename(columns={
                    "company_name": "Company",
                    "concept": "Concept",
                    "period_end": "Period End",
                    "quarter_label": "Quarter",
                    "value": "Value (USD)",
                    "unit": "Unit",
                })[["Company", "Concept", "Quarter", "Period End",
                    "Value (USD)", "Unit"]],
                use_container_width=True,
                hide_index=True,
            )


    main()
