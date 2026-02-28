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


def load_multi_facts(
    session_factory,
    ciks: list[str],
    concepts: list[str],
    start_date: date,
    end_date: date,
    forms: list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Load several XBRL concepts at once for the selected companies.

    Parameters:
        session_factory: SQLAlchemy sessionmaker.
        ciks:            List of CIKs to include.
        concepts:        XBRL concept names to load.
        start_date:      Earliest period_end to include.
        end_date:        Latest period_end to include.
        forms:           Filing form types to query.  Defaults to both
                         10-K and 10-Q.

    Returns:
        dict mapping concept name → DataFrame (same schema as
        :func:`load_facts`).  Concepts with no data are omitted.
    """
    forms_filter = forms or ["10-K", "10-Q"]
    result: dict[str, pd.DataFrame] = {}
    with session_factory() as session:
        fact_repo = FinancialFactRepository(session)
        company_repo = CompanyRepository(session)
        name_cache: dict[str, str] = {}

        for concept in concepts:
            rows: list[dict] = []
            for cik in ciks:
                if cik not in name_cache:
                    company = company_repo.get_by_cik(cik)
                    name_cache[cik] = company.name if company else cik
                for form in forms_filter:
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
                                "company_name": name_cache[cik],
                                "concept": f.concept,
                                "period_end": f.period_end,
                                "value": float(f.value),
                                "unit": f.unit,
                            }
                        )
            if not rows:
                continue
            df = pd.DataFrame(rows)
            df["period_end"] = pd.to_datetime(df["period_end"])
            df["quarter_label"] = (
                df["period_end"].dt.year.astype(str)
                + "-Q"
                + df["period_end"].dt.quarter.astype(str)
            )
            # De-duplicate: keep one row per (cik, period_end).
            df = df.sort_values("period_end").drop_duplicates(
                subset=["cik", "period_end"], keep="last"
            )
            result[concept] = df

    return result




_CONCEPT_LABELS: dict[str, str] = {
    "Revenues": "Revenue (USD)",
    "NetIncomeLoss": "Net Income / Loss (USD)",
    "Assets": "Total Assets (USD)",
    "Liabilities": "Total Liabilities (USD)",
    "DepreciationAndAmortization": "Depreciation & Amortization (USD)",
    "GainLossOnSaleOfProperties": "Gain / Loss on Property Sales (USD)",
    "GainsLossesOnSalesOfInvestmentRealEstate": "Gain / Loss on Real Estate Sales (USD)",
    "OperatingIncomeLoss": "Operating Income / Loss (USD)",
    "InterestExpense": "Interest Expense (USD)",
    "NetCashProvidedByUsedInOperatingActivities": "Operating Cash Flow (USD)",
    "RealEstateInvestmentPropertyNet": "Net Real Estate Value (USD)",
    "NumberOfRealEstateProperties": "Number of Properties",
    "AreaOfRealEstateProperty": "Leasable Area (sqft)",
    "CommonStockDividendsPerShareDeclared": "Dividends per Share (USD/share)",
    "LongTermDebt": "Long-term Debt (USD)",
    "LongTermDebtMaturitiesRepaymentsOfPrincipalInNextTwelveMonths": "Debt Due Y1 (USD)",
    "LongTermDebtMaturitiesRepaymentsOfPrincipalInYearTwo": "Debt Due Y2 (USD)",
    "LongTermDebtMaturitiesRepaymentsOfPrincipalInYearThree": "Debt Due Y3 (USD)",
    "LongTermDebtMaturitiesRepaymentsOfPrincipalInYearFour": "Debt Due Y4 (USD)",
    "LongTermDebtMaturitiesRepaymentsOfPrincipalInYearFive": "Debt Due Y5 (USD)",
    "LongTermDebtMaturitiesRepaymentsOfPrincipalAfterYearFive": "Debt Due After Y5 (USD)",
}

# Concepts shown in the "Financial Fundamentals" sidebar selector.
_SIDEBAR_CONCEPTS: list[str] = [
    "Revenues",
    "NetIncomeLoss",
    "Assets",
    "Liabilities",
    "DepreciationAndAmortization",
    "OperatingIncomeLoss",
    "InterestExpense",
    "NetCashProvidedByUsedInOperatingActivities",
    "RealEstateInvestmentPropertyNet",
    "LongTermDebt",
]


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


def build_ffo_chart(ffo_df: pd.DataFrame) -> alt.Chart:
    """Build a trend chart for computed FFO values."""
    return (
        alt.Chart(ffo_df)
        .mark_line(point=True)
        .encode(
            x=alt.X(
                "period_end:T",
                title="Period End",
                axis=alt.Axis(format="%b %Y", labelAngle=-45),
            ),
            y=alt.Y("ffo:Q", title="FFO (USD)", axis=alt.Axis(format="~s")),
            color=alt.Color("company_name:N", title="Company"),
            tooltip=[
                alt.Tooltip("company_name:N", title="Company"),
                alt.Tooltip("quarter_label:N", title="Quarter"),
                alt.Tooltip("ffo:Q", title="FFO (USD)", format=",.0f"),
            ],
        )
        .properties(title="Computed FFO – quarterly trend", height=380)
        .interactive()
    )


def build_generic_trend_chart(
    df: pd.DataFrame,
    concept: str,
    unit_label: str = "USD",
) -> alt.Chart:
    """Build a generic multi-line trend chart for any concept."""
    y_label = _CONCEPT_LABELS.get(concept, f"{concept} ({unit_label})")
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
                alt.Tooltip("value:Q", title=y_label, format=",.2f"),
            ],
        )
        .properties(title=y_label, height=300)
        .interactive()
    )


def build_debt_maturity_chart(maturity_df: pd.DataFrame) -> alt.Chart:
    """Build a grouped bar chart of debt maturities by year bucket."""
    return (
        alt.Chart(maturity_df)
        .mark_bar()
        .encode(
            x=alt.X("bucket:N", title="Maturity Bucket", sort=None),
            y=alt.Y("value:Q", title="Principal (USD)", axis=alt.Axis(format="~s")),
            color=alt.Color("company_name:N", title="Company"),
            xOffset="company_name:N",
            tooltip=[
                alt.Tooltip("company_name:N", title="Company"),
                alt.Tooltip("bucket:N", title="Maturity"),
                alt.Tooltip("value:Q", title="Principal (USD)", format=",.0f"),
            ],
        )
        .properties(title="Debt maturity schedule", height=350)
    )


# ---------------------------------------------------------------------------
# FFO / extended-metrics helpers
# ---------------------------------------------------------------------------

_MATURITY_BUCKET_MAP: dict[str, str] = {
    "LongTermDebtMaturitiesRepaymentsOfPrincipalInNextTwelveMonths": "Year 1",
    "LongTermDebtMaturitiesRepaymentsOfPrincipalInYearTwo": "Year 2",
    "LongTermDebtMaturitiesRepaymentsOfPrincipalInYearThree": "Year 3",
    "LongTermDebtMaturitiesRepaymentsOfPrincipalInYearFour": "Year 4",
    "LongTermDebtMaturitiesRepaymentsOfPrincipalInYearFive": "Year 5",
    "LongTermDebtMaturitiesRepaymentsOfPrincipalAfterYearFive": "After 5",
}


def compute_ffo(facts: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Compute FFO = NetIncomeLoss + D&A − GainLossOnSaleOfProperties.

    Merges the three component DataFrames on (cik, period_end).  Either
    gain-on-sale concept tag is accepted as the third component; when both
    are present the first non-null value per row is used.

    Parameters:
        facts: Dict from :func:`load_multi_facts`.

    Returns:
        pd.DataFrame with columns: cik, company_name, period_end,
        quarter_label, ffo.  Empty DataFrame when required inputs are
        missing.
    """
    ni = facts.get("NetIncomeLoss")
    da = facts.get("DepreciationAndAmortization")
    # Accept either gain-on-sale tag.
    gains = facts.get("GainLossOnSaleOfProperties") or facts.get(
        "GainsLossesOnSalesOfInvestmentRealEstate"
    )

    if ni is None or da is None:
        return pd.DataFrame()

    base_cols = ["cik", "company_name", "period_end", "quarter_label"]
    merged = ni[base_cols + ["value"]].rename(columns={"value": "net_income"})
    merged = merged.merge(
        da[base_cols + ["value"]].rename(columns={"value": "da"}),
        on=["cik", "period_end"],
        how="inner",
        suffixes=("", "_da"),
    )
    # Keep consistent company_name and quarter_label from the left side.
    merged = merged.drop(
        columns=[c for c in merged.columns if c.endswith("_da")]
    )

    if gains is not None:
        merged = merged.merge(
            gains[["cik", "period_end", "value"]].rename(
                columns={"value": "gains"}
            ),
            on=["cik", "period_end"],
            how="left",
        )
    else:
        merged["gains"] = 0.0

    merged["gains"] = merged["gains"].fillna(0.0)
    merged["ffo"] = merged["net_income"] + merged["da"] - merged["gains"]
    return merged[base_cols + ["ffo"]]


def build_maturity_df(
    facts: dict[str, pd.DataFrame],
    companies_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a tidy DataFrame of debt maturity buckets for charting.

    Uses the most recently reported maturity schedule per company.

    Parameters:
        facts:        Dict from :func:`load_multi_facts`.
        companies_df: Company metadata DataFrame.

    Returns:
        pd.DataFrame with columns: company_name, bucket, value.
    """
    cik_to_name = companies_df.set_index("cik")["name"].to_dict()
    rows: list[dict] = []
    for concept, label in _MATURITY_BUCKET_MAP.items():
        df = facts.get(concept)
        if df is None or df.empty:
            continue
        # Keep the most recent entry per company.
        latest = df.sort_values("period_end").groupby("cik").last().reset_index()
        for _, row in latest.iterrows():
            rows.append(
                {
                    "company_name": cik_to_name.get(row["cik"], row["cik"]),
                    "bucket": label,
                    "value": row["value"],
                }
            )
    return pd.DataFrame(rows)





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
        options=_SIDEBAR_CONCEPTS,
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
    # Section 3 – FFO Analysis                                           #
    # ================================================================== #
    st.header("📐 FFO Analysis (Computed)")
    st.caption(
        "FFO = Net Income + Depreciation & Amortisation − Gain on Property Sales.  "
        "Requires all three components to be reported by the company."
    )

    _FFO_CONCEPTS = [
        "NetIncomeLoss",
        "DepreciationAndAmortization",
        "GainLossOnSaleOfProperties",
        "GainsLossesOnSalesOfInvestmentRealEstate",
    ]
    ffo_facts = load_multi_facts(
        session_factory, selected_ciks, _FFO_CONCEPTS, start_date, end_date
    )
    ffo_df = compute_ffo(ffo_facts)

    if ffo_df.empty:
        st.info(
            "Insufficient data to compute FFO. "
            "Ensure Net Income and D&A data has been ingested."
        )
    else:
        # FFO metric cards.
        ffo_cols = st.columns(len(ffo_df["cik"].unique()))
        cik_to_name = companies_df.set_index("cik")["name"].to_dict()
        for col, cik in zip(ffo_cols, sorted(ffo_df["cik"].unique())):
            c_df = ffo_df[ffo_df["cik"] == cik].sort_values("period_end")
            if c_df.empty:
                continue
            latest_ffo = c_df.iloc[-1]["ffo"]
            latest_q = c_df.iloc[-1]["quarter_label"]
            one_year_ago = c_df.iloc[-1]["period_end"] - pd.DateOffset(years=1)
            prior = c_df[
                (c_df["period_end"] >= one_year_ago - pd.Timedelta(days=45))
                & (c_df["period_end"] <= one_year_ago + pd.Timedelta(days=45))
            ]
            ffo_delta: str | None = None
            if not prior.empty and prior.iloc[-1]["ffo"] != 0:
                pct = (latest_ffo - prior.iloc[-1]["ffo"]) / abs(prior.iloc[-1]["ffo"]) * 100
                ffo_delta = f"{pct:+.1f}% YoY"
            col.metric(
                label=cik_to_name.get(cik, cik),
                value=f"${latest_ffo / 1e6:,.1f}M",
                delta=ffo_delta,
                help=f"Latest: {latest_q}",
            )

        st.altair_chart(build_ffo_chart(ffo_df), use_container_width=True)

        with st.expander("FFO Component Data"):
            comp_dfs = []
            for c_name in ["NetIncomeLoss", "DepreciationAndAmortization",
                           "GainLossOnSaleOfProperties",
                           "GainsLossesOnSalesOfInvestmentRealEstate"]:
                if c_name in ffo_facts:
                    d = ffo_facts[c_name].copy()
                    d["component"] = _CONCEPT_LABELS.get(c_name, c_name)
                    comp_dfs.append(d)
            if comp_dfs:
                comp_df = pd.concat(comp_dfs)
                st.dataframe(
                    comp_df[["company_name", "component", "quarter_label", "value"]]
                    .rename(columns={
                        "company_name": "Company",
                        "component": "Component",
                        "quarter_label": "Quarter",
                        "value": "Value (USD)",
                    }),
                    use_container_width=True,
                    hide_index=True,
                )

    # ================================================================== #
    # Section 4 – Portfolio Metrics                                       #
    # ================================================================== #
    st.header("🏗️ Portfolio Metrics")

    _PORTFOLIO_CONCEPTS = [
        "NumberOfRealEstateProperties",
        "AreaOfRealEstateProperty",
        "RealEstateInvestmentPropertyNet",
    ]
    portfolio_facts = load_multi_facts(
        session_factory,
        selected_ciks,
        _PORTFOLIO_CONCEPTS,
        start_date,
        end_date,
        forms=["10-K", "10-Q"],
    )

    if not portfolio_facts:
        st.info("No portfolio metric data available. Re-run ingestion to fetch extended EDGAR data.")
    else:
        tab_props, tab_sqft, tab_value = st.tabs(
            ["🏢 Property Count", "📐 Leasable Area", "💰 Net Property Value"]
        )
        with tab_props:
            df_props = portfolio_facts.get("NumberOfRealEstateProperties")
            if df_props is not None and not df_props.empty:
                st.altair_chart(
                    build_generic_trend_chart(df_props, "NumberOfRealEstateProperties", "properties"),
                    use_container_width=True,
                )
            else:
                st.info("No property count data available.")
        with tab_sqft:
            df_sqft = portfolio_facts.get("AreaOfRealEstateProperty")
            if df_sqft is not None and not df_sqft.empty:
                st.altair_chart(
                    build_generic_trend_chart(df_sqft, "AreaOfRealEstateProperty", "sqft"),
                    use_container_width=True,
                )
            else:
                st.info("No leasable area data available.")
        with tab_value:
            df_val = portfolio_facts.get("RealEstateInvestmentPropertyNet")
            if df_val is not None and not df_val.empty:
                st.altair_chart(
                    build_generic_trend_chart(df_val, "RealEstateInvestmentPropertyNet"),
                    use_container_width=True,
                )
            else:
                st.info("No net property value data available.")

    # ================================================================== #
    # Section 5 – Dividends                                              #
    # ================================================================== #
    st.header("💰 Dividends per Share")

    div_facts = load_multi_facts(
        session_factory,
        selected_ciks,
        ["CommonStockDividendsPerShareDeclared"],
        start_date,
        end_date,
    )
    div_df = div_facts.get("CommonStockDividendsPerShareDeclared")
    if div_df is None or div_df.empty:
        st.info("No dividend data available. Re-run ingestion to fetch extended EDGAR data.")
    else:
        # Metric cards: latest dividend and YoY.
        div_cols = st.columns(len(div_df["cik"].unique()))
        cik_to_name = companies_df.set_index("cik")["name"].to_dict()
        for col, cik in zip(div_cols, sorted(div_df["cik"].unique())):
            c_df = div_df[div_df["cik"] == cik].sort_values("period_end")
            if c_df.empty:
                continue
            latest_div = c_df.iloc[-1]["value"]
            latest_q = c_df.iloc[-1]["quarter_label"]
            one_year_ago = c_df.iloc[-1]["period_end"] - pd.DateOffset(years=1)
            prior = c_df[
                (c_df["period_end"] >= one_year_ago - pd.Timedelta(days=45))
                & (c_df["period_end"] <= one_year_ago + pd.Timedelta(days=45))
            ]
            div_delta: str | None = None
            if not prior.empty and prior.iloc[-1]["value"] != 0:
                pct = (latest_div - prior.iloc[-1]["value"]) / abs(prior.iloc[-1]["value"]) * 100
                div_delta = f"{pct:+.1f}% YoY"
            col.metric(
                label=cik_to_name.get(cik, cik),
                value=f"${latest_div:.4f}/share",
                delta=div_delta,
                help=f"Latest: {latest_q}",
            )
        st.altair_chart(
            build_generic_trend_chart(
                div_df, "CommonStockDividendsPerShareDeclared", "USD/share"
            ),
            use_container_width=True,
        )

    # ================================================================== #
    # Section 6 – Debt Profile                                           #
    # ================================================================== #
    st.header("🏦 Debt Profile")

    _DEBT_CONCEPTS = [
        "LongTermDebt",
        "LongTermDebtMaturitiesRepaymentsOfPrincipalInNextTwelveMonths",
        "LongTermDebtMaturitiesRepaymentsOfPrincipalInYearTwo",
        "LongTermDebtMaturitiesRepaymentsOfPrincipalInYearThree",
        "LongTermDebtMaturitiesRepaymentsOfPrincipalInYearFour",
        "LongTermDebtMaturitiesRepaymentsOfPrincipalInYearFive",
        "LongTermDebtMaturitiesRepaymentsOfPrincipalAfterYearFive",
    ]
    debt_facts = load_multi_facts(
        session_factory,
        selected_ciks,
        _DEBT_CONCEPTS,
        start_date,
        end_date,
        forms=["10-K", "10-Q"],
    )

    if not debt_facts:
        st.info("No debt data available. Re-run ingestion to fetch extended EDGAR data.")
    else:
        tab_trend, tab_maturity = st.tabs(
            ["📉 Long-term Debt Trend", "📅 Maturity Schedule"]
        )
        with tab_trend:
            ltd_df = debt_facts.get("LongTermDebt")
            if ltd_df is not None and not ltd_df.empty:
                st.altair_chart(
                    build_generic_trend_chart(ltd_df, "LongTermDebt"),
                    use_container_width=True,
                )
            else:
                st.info("No long-term debt data available.")
        with tab_maturity:
            mat_df = build_maturity_df(debt_facts, companies_df)
            if mat_df.empty:
                st.info("No debt maturity data available (annual 10-K filings required).")
            else:
                st.altair_chart(
                    build_debt_maturity_chart(mat_df),
                    use_container_width=True,
                )
                st.caption(
                    "Maturity schedule based on most recently filed 10-K disclosures."
                )


if __name__ == "__main__":
    main()
