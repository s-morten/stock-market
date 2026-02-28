# REIT Dashboard

A Python-based analytical dashboard for US REIT (Real Estate Investment Trust)
financial data, sourced from the **SEC EDGAR** public API.

## Architecture

```
src/reit_dashboard/
├── config.py               # Environment / settings loader
├── data/
│   ├── edgar_client.py     # SEC EDGAR REST API client (httpx + Pydantic)
│   ├── models.py           # SQLAlchemy ORM models (Company, FinancialFact)
│   ├── repository.py       # Database CRUD abstraction layer
│   └── ingestion.py        # Orchestration: fetch → validate → persist
└── dashboard/
    └── app.py              # Streamlit dashboard UI
scripts/
└── ingest_poc.py           # CLI: seed the database with PoC REIT data
tests/
└── data/                   # Unit tests (pytest + respx for HTTP mocking)
```

## Setup

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. Install the package with development dependencies
pip install -e ".[dev]"

# 3. Configure environment variables
cp .env.example .env
# Edit .env and set EDGAR_USER_AGENT to identify your application.
```

## Running the PoC Ingestion

Fetches financial data for 5 well-known REITs from EDGAR and stores it
in a local SQLite database:

```bash
python scripts/ingest_poc.py
```

Output example:
```
Company                        Facts   Concepts  Status
------------------------------------------------------------
Prologis                         120          4  OK
Realty Income                     98          4  OK
...
```

## Running the Dashboard

```bash
streamlit run src/reit_dashboard/dashboard/app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.
Use the sidebar to filter by company, financial concept, filing type, and date range.

## Running Tests

```bash
pytest tests/ -v
```

## Switching to Oracle

Update `DATABASE_URL` in your `.env` file:

```
DATABASE_URL=oracle+oracledb://user:password@host:1521/service_name
```

Also install the Oracle dialect driver:

```bash
pip install oracledb
```

No code changes are required.

## PoC REIT Universe

| Company | CIK |
|---|---|
| Prologis | 0001045609 |
| Realty Income | 0000726854 |
| Simon Property Group | 0001063761 |
| Public Storage | 0001393311 |
| Welltower | 0000766704 |

## Financial Concepts Fetched

| Dashboard Label | XBRL Tag |
|---|---|
| Revenue | `Revenues` |
| Net Income | `NetIncomeLoss` |
| Total Assets | `Assets` |
| Total Liabilities | `Liabilities` |

## Data Source

All data is fetched from the **SEC EDGAR** public REST API
(`https://data.sec.gov`).  The SEC requires that all automated clients
identify themselves via a `User-Agent` header – set this in `.env`.
