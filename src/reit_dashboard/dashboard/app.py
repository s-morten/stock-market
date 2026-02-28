"""
Streamlit dashboard for REIT financial data analysis.

Run with:
    streamlit run src/reit_dashboard/dashboard/app.py

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
        pd.DataFrame: Columns cik, name, sic, fiscal_year_end.
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

    Parameters:
        session_factory: SQLAlchemy sessionmaker.
        ciks:            List of CIKs to include.
        concept:         XBRL concept name.
        form:            Filing form type.
        start_date:      Earliest period_end to include.
        end_date:        Latest period_end to include.

    Returns:
        pd.DataFrame: Columns cik, company_name, concept, period_end, value, unit.
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
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Chart helper
# ---------------------------------------------------------------------------


def build_time_series_chart(df: pd.DataFrame, concept: str) -> alt.Chart:
    """
    Build an Altair line chart for the given financial concept data.

    Parameters:
        df:      DataFrame with columns company_name, period_end, value.
        concept: Label used in the chart title and Y-axis.

    Returns:
        alt.Chart: Interactive line chart.
    """
    chart = (
        alt.Chart(df)
        .mark_line(point=True)
        .encode(
            x=alt.X("period_end:T", title="Period End"),
            y=alt.Y("value:Q", title=f"{concept} (USD)"),
            color=alt.Color("company_name:N", title="Company"),
            tooltip=["company_name", "period_end", "value", "unit"],
        )
        .properties(title=f"{concept} over time", width=700, height=350)
        .interactive()
    )
    return chart


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
    st.caption("Data sourced from SEC EDGAR · PoC – 5 companies")

    session_factory = get_session_factory()
    companies_df = load_companies(session_factory)

    if companies_df.empty:
        st.warning(
            "No data found. Run `python scripts/ingest_poc.py` first to "
            "populate the database."
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
    )

    form = st.sidebar.radio("Filing Type", options=["10-K", "10-Q"], index=0)

    default_start = date.today() - timedelta(days=365 * 5)
    start_date = st.sidebar.date_input("Start Date", value=default_start)
    end_date = st.sidebar.date_input("End Date", value=date.today())

    # --- Company info table ---
    st.subheader("Company Overview")
    display_companies = companies_df[companies_df["cik"].isin(selected_ciks)]
    st.dataframe(display_companies.rename(columns={
        "cik": "CIK", "name": "Name", "sic": "SIC", "fiscal_year_end": "FY End"
    }), use_container_width=True, hide_index=True)

    if not selected_ciks:
        st.info("Select at least one company in the sidebar.")
        return

    # --- Financial facts ---
    facts_df = load_facts(
        session_factory, selected_ciks, concept, form, start_date, end_date
    )

    if facts_df.empty:
        st.warning(
            f"No {form} data found for **{concept}** in the selected period. "
            "Try adjusting the filters or re-running ingestion."
        )
        return

    st.subheader(f"{concept} – {form} filings")
    chart = build_time_series_chart(facts_df, concept)
    st.altair_chart(chart, use_container_width=True)

    with st.expander("Raw Data"):
        st.dataframe(
            facts_df.rename(columns={
                "cik": "CIK",
                "company_name": "Company",
                "concept": "Concept",
                "period_end": "Period End",
                "value": "Value (USD)",
                "unit": "Unit",
            }),
            use_container_width=True,
            hide_index=True,
        )


if __name__ == "__main__":
    main()
