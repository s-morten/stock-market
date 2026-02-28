"""
Configuration module for REIT Dashboard.

Loads environment variables from a .env file (if present) and exposes
typed settings used across the application.
"""

import os

from dotenv import load_dotenv

# Load .env file values into the environment (no-op if file is absent).
load_dotenv()


def get_database_url() -> str:
    """
    Return the SQLAlchemy database connection URL.

    Reads the DATABASE_URL environment variable.  Defaults to a local
    SQLite file so the project works out-of-the-box without any setup.
    Switching to Oracle requires only updating this env var – no code
    changes are needed.

    Returns:
        str: A valid SQLAlchemy connection string.
    """
    return os.getenv("DATABASE_URL", "sqlite:///./reit_data.db")


def get_edgar_user_agent() -> str:
    """
    Return the User-Agent string required by the SEC EDGAR API.

    The SEC requires every automated client to identify itself via a
    descriptive User-Agent header (see https://www.sec.gov/developer).

    Returns:
        str: User-Agent value, e.g. "MyApp contact@example.com".

    Raises:
        EnvironmentError: If the variable is not set.
    """
    value = os.getenv("EDGAR_USER_AGENT")
    if not value:
        raise EnvironmentError(
            "EDGAR_USER_AGENT environment variable is not set. "
            "Please copy .env.example to .env and fill in your details."
        )
    return value
